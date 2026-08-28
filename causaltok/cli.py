from __future__ import annotations
import argparse
from .world import FiniteWorld
from .public_worlds import PUBLIC_WORLDS


def main(argv=None):
    parser = argparse.ArgumentParser(prog="causaltok")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list-worlds")

    exp = sub.add_parser("export-world")
    exp.add_argument("name", choices=sorted(PUBLIC_WORLDS))
    exp.add_argument("--out", required=True)
    exp.add_argument("--size", type=int, default=None, help="family-specific scale parameter")

    ins = sub.add_parser("inspect")
    ins.add_argument("path")

    args = parser.parse_args(argv)
    if args.cmd == "list-worlds":
        for name in sorted(PUBLIC_WORLDS):
            print(name)
        return

    if args.cmd == "inspect":
        world = FiniteWorld.from_json(args.path)
        print(f"states={world.n_states} actions={world.n_actions} probabilities={world.probabilities is not None}")
        return

    factory = PUBLIC_WORLDS[args.name]
    if args.size is None:
        world = factory()
    elif args.name == "delayed":
        world = factory(depth=args.size)
    elif args.name == "duplicate":
        world = factory(causal_states=max(1, args.size // 20), duplicates=20)
    else:
        world = factory(duplicates=args.size)
    world.to_json(args.out)
    print(f"wrote {args.out}: states={world.n_states} actions={world.n_actions}")


if __name__ == "__main__":
    main()
