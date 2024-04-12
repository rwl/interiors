# interiors

The `interiors` crate solves non-linear programming problems (NLPs) using a
primal-dual interior point method. It is based on the [MATPOWER Interior Point Solver (MIPS)][1].
MIPS is based on [code written in C language][2] by Hongye Wang as a graduate
student at Cornell University for optimal power flow applications. It was
later ported to the MATLAB/Octave language by Ray D. Zimmerman for use in [MATPOWER][3].

## [Citation](CITATION)

We request that publications derived from the use of `interiors` explicitly
acknowledge the MATPOWER Interior Point Solver (MIPS) by citing the
2007 paper:

> H. Wang, C. E. Murillo-Sánchez, R. D. Zimmerman, R. J. Thomas, "On
> Computational Issues of Market-Based Optimal Power Flow," *Power Systems,
IEEE Transactions on*, vol. 22, no. 3, pp. 1185-1193, Aug. 2007.
> doi: [10.1109/TPWRS.2007.901301][4]

## License

The source code for `interiors` is distributed under the [3-clause BSD license](LICENSE).

[1]: https://github.com/MATPOWER/mips

[2]: http://www.pserc.cornell.edu/tspopf/

[3]: https://github.com/MATPOWER/matpower

[4]: https://doi.org/10.1109/TPWRS.2007.901301