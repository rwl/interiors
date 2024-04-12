# interiors

The `interiors` crate solves non-linear programming problems (NLPs) using a
primal-dual interior point method. It is based on the [MATPOWER Interior Point Solver (MIPS)][1].
MIPS is based on [code written in C language][2] by Hongye Wang as a graduate
student at Cornell University for optimal power flow applications. It was
later ported to the MATLAB/Octave language by Ray D. Zimmerman for use in [MATPOWER][3].

## Citation

We request that publications derived from the use of `interiors` explicitly
acknowledge the MATPOWER Interior Point Solver (MIPS) by citing the
[2007 paper](CITATION).

## License

The source code is distributed under the [3-clause BSD license](LICENSE).

[1]: https://github.com/MATPOWER/mips

[2]: http://www.pserc.cornell.edu/tspopf/

[3]: https://github.com/MATPOWER/matpower