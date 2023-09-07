use crate::LinearSolver;
use anyhow::{format_err, Result};
use sparsetools::csr::CSR;

pub struct RLUSolver {
    pub amd_control: amd::Control,
    pub rlu_options: rlu::Options,
}

impl LinearSolver for RLUSolver {
    fn solve(&self, a_mat: &CSR<usize, f64>, b: &mut [f64]) -> Result<()> {
        let n = a_mat.cols();
        // let a_p = a_mat.indptr().raw_storage();
        // let a_p = a_mat.indptr().as_slice().unwrap();
        let a_p = a_mat.rowptr();
        let a_i = a_mat.colidx();
        let a = a_mat.data();

        let col_perm = match amd::order::<usize>(n, &a_p, &a_i, &self.amd_control) {
            Err(status) => return Err(format_err!("amd error: {:?}", status)),
            Ok((p, _p_inv, _info)) => p,
        };

        let lu = rlu::factor::<usize, f64>(n, &a_i, &a_p, &a, Some(&col_perm), &self.rlu_options)
            .map_err(|err| format_err!("factor error: {}", err))?;

        rlu::solve(&lu, b, false).map_err(|err| format_err!("solve error: {}", err))
    }
}

impl Default for RLUSolver {
    fn default() -> Self {
        Self {
            amd_control: amd::Control::default(),
            rlu_options: rlu::Options::default(),
        }
    }
}
