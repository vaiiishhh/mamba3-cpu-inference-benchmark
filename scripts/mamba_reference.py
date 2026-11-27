#!/usr/bin/env python3
"""Generate a streamed FP32 CPU PyTorch reference trace for mamba_decode."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
from typing import Optional

import torch
import torch.nn.functional as F


# References at this state-spaces/mamba revision:
# mamba_ssm/modules/mamba3.py (mixer preprocessing and decode ordering)
# mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py (eager rotary reference)
# mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py (eager recurrence reference)
# mamba_ssm/modules/mamba2.py (eager Mamba-2 recurrence)
UPSTREAM = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
PROJECTIONS = ("in_proj.weight", "out_proj.weight")
MAMBA3_PARAMS = ("dt_bias", "B_norm.weight", "C_norm.weight", "B_bias", "C_bias",
                 "D", "mimo_x", "mimo_z", "mimo_o")
CHECKPOINTS = {
    "siso": {
        "model": "state-spaces/mamba3-siso-187m",
        "revision": "6792c27c00f3bb41506db1066dcd1c51bb0f4b02",
        "sha256": "b462d55e9f8ab9746be641d4de550b51bf65e55ab662e66835588b5a458c95ec",
    },
    "mimo": {
        "model": "state-spaces/mamba3-mimo-187m",
        "revision": "8fd6e9eb7b795f2e15d7f6353251d0137980c43e",
        "sha256": "369db3fb9deedfc98baa95ac9166baf862dd7e03b3143afaf0d19e46a220daf6",
    },
}


def load_official_eager(source):
    """Extract only eager Torch functions from the pinned GPU modules."""
    functions = []
    for name, relative_path, digest in (
        ("apply_rotary_qk_inference_reference",
         "ops/triton/mamba3/mamba3_mimo_rotary_step.py",
         "e117b5da3d2ddfcfa66673a88dfd88a6d688100291d3b6a1b0b3c24c82ef79d6"),
        ("selective_state_update_fused_ref_v2",
         "ops/cute/mamba3/mamba3_step_fn.py",
         "5c82f3936308cfc90bb3bcdfd410c60fc80dfc3afe72c4338831341398cd7ca4"),
    ):
        path = source / "mamba_ssm" / relative_path
        code = path.read_bytes()
        if hashlib.sha256(code).hexdigest() != digest:
            raise ValueError(f"{path} does not match state-spaces/mamba at {UPSTREAM}")
        tree = ast.parse(code, filename=str(path))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = {"torch": torch, "math": math, "Optional": Optional}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        functions.append(namespace[name])
    return functions


class Reference:
    def __init__(self, args):
        self.args = args
        self.model = 2 if args.model == "mamba2" else 3
        shape = (24, 64, 128, 1) if args.shape == "checkpoint" else (4, 5, 10, 2)
        self.h, self.p, self.n, self.g = shape
        self.r = 4 if args.model == "mimo" else 1
        self.k = self.n // 4 if self.model == 3 else 0
        if self.model == 3 and args.rotary == "full":
            self.k = self.n // 2
        if args.rotary == "none":
            self.k = 0
        self.mixer_width = 768 if args.shape == "checkpoint" else 10
        self.d = self.mixer_width if args.mode == "mixer" else 0
        self.hp = self.h * self.p
        self.bc = self.r * self.g * self.n
        self.parts = [self.hp, self.hp, self.bc, self.bc, self.h, self.h]
        if self.model == 3:
            self.parts += [self.h, self.k]
        self.group = torch.arange(self.h) // (self.h // self.g)
        self.rng = torch.Generator(device="cpu").manual_seed(args.seed)
        self.official_functions = load_official_eager(args.upstream_source) if args.upstream_source else None
        self.weights = self.make_weights()
        self.state = [torch.zeros(self.h, self.p, self.n)]
        if self.model == 3:
            self.state += [
                torch.zeros(self.h, self.k),
                torch.zeros(self.h, self.r, self.n),
                torch.zeros(self.h, self.p),
            ]
        if args.initial == "nonzero":
            for index in range(17):
                self.advance(self.token(index))

    def rand(self, *shape, scale=1.0):
        return torch.randn(shape, generator=self.rng, dtype=torch.float32) * scale

    def make_weights(self):
        if self.args.checkpoint:
            return self.load_mixer(self.args.checkpoint)
        if self.model == 2:
            return {"D": 0.9 + self.rand(self.h, scale=0.13)}
        weights = {}
        if self.d:
            weights["in_proj.weight"] = self.rand(sum(self.parts), self.d, scale=0.23 / math.sqrt(self.d))
            weights["out_proj.weight"] = self.rand(self.d, self.hp, scale=0.4 / math.sqrt(self.hp))
        weights.update({
            "dt_bias": -2.8 + self.rand(self.h, scale=0.15),
            "B_norm.weight": 1 + self.rand(self.n, scale=0.12),
            "C_norm.weight": 1 + self.rand(self.n, scale=0.12),
            "B_bias": 1 + self.rand(self.h, self.r, self.n, scale=0.16),
            "C_bias": 1 + self.rand(self.h, self.r, self.n, scale=0.16),
            "D": 1 + self.rand(self.h, scale=0.15),
        })
        for name in ("mimo_x", "mimo_z", "mimo_o"):
            shape = (self.h, self.r, self.p)
            weights[name] = torch.ones(shape) if self.r == 1 else 0.55 + self.rand(*shape, scale=0.27)
        if self.args.scenario == "one-active-rank":
            weights["mimo_x"][:, 1:].zero_()
            weights["mimo_o"][:, 1:].zero_()
        if self.args.scenario == "impulse":
            weights["B_norm.weight"].zero_()
            weights["C_norm.weight"].zero_()
            weights["B_bias"].fill_(1)
            weights["C_bias"].fill_(1)
            weights["dt_bias"].zero_()
            weights["D"].zero_()
            for name in ("mimo_x", "mimo_z", "mimo_o"):
                weights[name].fill_(1)
        return weights

    def load_mixer(self, path):
        official = CHECKPOINTS[self.args.model]
        with path.open("rb") as file:
            checksum = hashlib.file_digest(file, "sha256").hexdigest()
        if checksum != official["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {official['model']} at {official['revision']}")
        # Map storage and materialize only the selected mixer in FP32.
        model = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        prefix = f"backbone.layers.{self.args.layer}.mixer."
        shapes = {
            "in_proj.weight": (sum(self.parts), self.mixer_width),
            "out_proj.weight": (self.mixer_width, self.hp),
            "dt_bias": (self.h,),
            "B_norm.weight": (self.n,),
            "C_norm.weight": (self.n,),
            "B_bias": (self.h, self.r, self.n),
            "C_bias": (self.h, self.r, self.n),
            "D": (self.h,),
        }
        if self.r == 4:
            shapes.update(dict.fromkeys(("mimo_x", "mimo_z", "mimo_o"), (self.h, self.r, self.p)))
        weights = {}
        for name, shape in shapes.items():
            tensor = model[prefix + name]
            if tensor.shape != shape:
                raise ValueError(f"{prefix}{name} has shape {tensor.shape}, expected {shape}")
            weights[name] = tensor.float()
        if self.r == 1:
            weights.update({name: torch.ones(self.h, 1, self.p) for name in ("mimo_x", "mimo_z", "mimo_o")})
        return weights

    def token(self, step):
        if self.d:
            return self.rand(self.d, scale=0.35)
        if self.args.checkpoint:
            hidden = self.rand(self.mixer_width, scale=0.35)
            return F.linear(hidden, self.weights["in_proj.weight"])
        z = self.rand(self.hp, scale=0.65)
        x = self.rand(self.hp, scale=0.32)
        b = self.rand(self.bc, scale=0.7)
        c = self.rand(self.bc, scale=0.7)
        if self.model == 2:
            dt = F.softplus(-2.8 + self.rand(self.h, scale=0.3))
            decay = -torch.exp(self.rand(self.h, scale=0.3))
            return torch.cat([z, x, b, c, dt, decay])
        dt = self.rand(self.h, scale=0.3)
        decay = self.rand(self.h, scale=0.6)
        mix = self.rand(self.h, scale=0.7)
        angle = self.rand(self.k, scale=0.5)
        if self.args.scenario == "zero-angle":
            angle.zero_()
        elif self.args.scenario == "trap-extremes":
            mix[::2] = 24 if step % 2 else -24
            mix[1::2] = -24 if step % 2 else 24
        elif self.args.scenario == "impulse":
            z.fill_(1)
            x.fill_(int(step == 0))
            b.zero_()
            c.zero_()
            dt.zero_()
            decay.zero_()
            mix.zero_()
            angle.zero_()
        return torch.cat([z, x, b, c, dt, decay, mix, angle])

    def rotate(self, field, cos, sin):
        # The eager upstream reference pairs adjacent coordinates for SISO and
        # pairs the corresponding entries of the two halves for MIMO.
        if not self.k:
            return field
        if self.r == 1:
            pairs = field[..., : self.k * 2].reshape(self.h, self.r, self.k, 2)
            a, b = pairs.unbind(-1)
            rotated = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)
            return torch.cat([rotated, field[..., self.k * 2 :]], dim=-1)
        a, b = field.chunk(2, dim=-1)
        a_rot, b_rot = a[..., : self.k], b[..., : self.k]
        return torch.cat([a_rot * cos - b_rot * sin, a[..., self.k :],
                          a_rot * sin + b_rot * cos, b[..., self.k :]], dim=-1)

    def advance(self, data):
        w = self.weights
        projected = F.linear(data, w["in_proj.weight"]) if self.d else data
        z, x, b, c, dt, decay, *extras = projected.split(self.parts)
        x, z = x.reshape(self.h, self.p), z.reshape(self.h, self.p)
        b, c = (t.reshape(self.r, self.g, self.n) for t in (b, c))
        if self.model == 2:
            # Mamba2.step's eager branch, generalized to grouped B and C.
            b, c = b[0, self.group], c[0, self.group]
            state = self.state[0] * (dt * decay).exp()[:, None, None]
            state += torch.einsum("h,hn,hp->hpn", dt, b, x)
            self.state = [state]
            return (torch.einsum("hpn,hn->hp", state, c) + w["D"][:, None] * x) * F.silu(z)
        dt = F.softplus(dt + w["dt_bias"])
        # Mamba3's current data-dependent decay uses the heavy-tail activation.
        decay = -(decay.clamp_min(0) + (1 - decay.clamp_max(0)).reciprocal()).clamp_min(1e-4)
        mix, angle = extras
        mix = mix.sigmoid()
        state, phase, prev_b, prev_x = self.state
        b = F.rms_norm(b, (self.n,), w["B_norm.weight"], eps=1e-5)
        c = F.rms_norm(c, (self.n,), w["C_norm.weight"], eps=1e-5)
        if self.official_functions:
            rotate, update = self.official_functions
            c, b, phase = rotate(c[:, self.group][None], b[:, self.group][None], phase[None],
                                 angle.expand(1, self.h, self.k), dt[None],
                                 bias_q=w["C_bias"].transpose(0, 1), bias_k=w["B_bias"].transpose(0, 1),
                                 rotate_pairwise=self.r == 1)
            result, state = update(state[None], decay[None], b, c,
                                   w["mimo_x"].transpose(0, 1), x[None],
                                   w["mimo_z"].transpose(0, 1), z[None], dt[None],
                                   prev_b.transpose(0, 1)[None], prev_x[None], mix[None],
                                   w["D"], w["mimo_o"].transpose(0, 1))
            self.state = [state[0], phase[0], b[0].transpose(0, 1), x]
            return F.linear(result.flatten(), w["out_proj.weight"]) if self.d else result[0]

        phase = phase + angle.tanh() * dt[:, None] * math.pi
        cos, sin = phase.cos().unsqueeze(1), phase.sin().unsqueeze(1)
        b = self.rotate(b[:, self.group].transpose(0, 1) + w["B_bias"], cos, sin)
        c = self.rotate(c[:, self.group].transpose(0, 1) + w["C_bias"], cos, sin)

        # The official eager recurrence expressed in the fixture's [H,R,*] layout.
        alpha = (decay * dt).exp()
        beta = (1 - mix) * dt * alpha
        gamma = mix * dt
        v = x[:, None, :] * w["mimo_x"]
        prev_v = prev_x[:, None, :] * w["mimo_x"]
        current = torch.einsum("hrp,hrn->hpn", v * gamma[:, None, None], b)
        previous = torch.einsum("hrp,hrn->hpn", prev_v * beta[:, None, None], prev_b)
        state = state * alpha[:, None, None] + current + previous
        readout = torch.einsum("hpn,hrn->hrp", state, c)
        readout += w["D"][:, None, None] * v
        readout *= F.silu(z[:, None, :] * w["mimo_z"])
        result = (readout * w["mimo_o"]).sum(dim=1)
        self.state = [state, phase, b, x]
        return F.linear(result.flatten(), w["out_proj.weight"]) if self.d else result


def write_tensors(file, tensors):
    for tensor in tensors:
        file.write(tensor.contiguous().numpy().astype("<f4", copy=False).tobytes())


@torch.inference_mode()
def generate(args):
    torch.set_num_threads(1)
    reference = Reference(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=args.output.name + ".", delete=False) as file:
            temp_path = Path(file.name)
            file.write(struct.pack("<8s9I", b"MAMBA3V1", reference.model, reference.h, reference.p, reference.n,
                                   reference.r, reference.g, reference.k, reference.d, args.steps))
            weights = MAMBA3_PARAMS if reference.model == 3 else ("D",)
            if reference.d:
                weights = PROJECTIONS + weights
            write_tensors(file, (reference.weights[name] for name in weights))
            write_tensors(file, reference.state)
            for step in range(args.steps):
                data = reference.token(step)
                output = reference.advance(data)
                if args.scenario == "impulse" and step in (0, 1):
                    # dt=log(2), A=-1 gives alpha=1/2 and mix=1/2.
                    # On step two both the decayed state and previous input contribute.
                    expected = reference.r * math.log(2) / 2
                    torch.testing.assert_close(reference.state[0], torch.full_like(reference.state[0], expected))
                write_tensors(file, [data, output, *reference.state])
        temp_path.replace(args.output)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    metadata = {
        "fixture_format": "MAMBA3V1",
        "reference_commit": UPSTREAM,
        "reference_backend": "upstream_eager" if args.upstream_source else "local_eager",
        "torch": torch.__version__,
        "seed": args.seed,
        "model": args.model,
        "mode": args.mode,
        "scenario": args.scenario,
        "initial": args.initial,
        "dimensions": dict(H=reference.h, P=reference.p, N=reference.n, R=reference.r,
                           groups=reference.g, angle_pairs=reference.k, mixer_width=reference.d),
        "steps": args.steps,
        "bytes": args.output.stat().st_size,
    }
    if args.checkpoint:
        metadata.update(CHECKPOINTS[args.model], variant=args.model, layer=args.layer)
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("mamba2", "siso", "mimo"), required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--shape", choices=("small", "checkpoint"), default="small")
    parser.add_argument("--mode", choices=("operator", "mixer"), default="operator")
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--initial", choices=("zero", "nonzero"), default="zero")
    parser.add_argument("--scenario", choices=("random", "impulse", "zero-angle", "trap-extremes",
                                               "one-active-rank"), default="random")
    parser.add_argument("--rotary", choices=("half", "full", "none"), default="half")
    parser.add_argument("--checkpoint", type=Path, help="Pinned official pytorch_model.bin for this variant")
    parser.add_argument("--upstream-source", type=Path,
                        help=f"Optional state-spaces/mamba checkout at {UPSTREAM} for direct eager execution")
    parser.add_argument("--layer", type=int, default=0, help="Checkpoint mixer layer, from 0 to 11")
    args = parser.parse_args()
    if not 1 <= args.steps < 2**32:
        parser.error("--steps must be in [1, 2^32)")
    if args.model == "mamba2" and args.mode != "operator":
        parser.error("Mamba-2 supports operator fixtures")
    if args.model == "mamba2" and args.upstream_source:
        parser.error("direct upstream extraction is available for Mamba-3")
    if args.checkpoint and (args.model == "mamba2" or args.shape != "checkpoint"
                            or args.rotary != "half" or args.scenario != "random" or not 0 <= args.layer < 12):
        parser.error("--checkpoint requires Mamba-3 with its checkpoint shape and layer 0 through 11")
    if args.scenario != "random" and (args.mode != "operator" or args.model == "mamba2"):
        parser.error("special scenarios require a Mamba-3 operator fixture")
    if args.scenario == "one-active-rank" and args.model != "mimo":
        parser.error("one-active-rank requires MIMO")
    if args.scenario == "impulse" and (args.initial != "zero" or args.steps < 2):
        parser.error("impulse requires zero initial state and at least two steps")
    generate(args)


if __name__ == "__main__":
    main()
