use crate::LinearSolver;
use ndarray::ArrayViewMut1;
use sprs::CsMatView;

pub struct RLUSolver {
    pub amd_control: amd::Control,
    pub rlu_options: rlu::Options,
}

impl LinearSolver for RLUSolver {
    fn solve(&self, a_mat: CsMatView<f64>, mut b: ArrayViewMut1<f64>) -> Result<(), String> {
        let n = a_mat.cols();
        // let a_p = a_mat.indptr().raw_storage();
        // let a_p = a_mat.indptr().as_slice().unwrap();
        let a_p = a_mat.indptr();
        let a_i = a_mat.indices();
        let a = a_mat.data();

        let col_perm = match amd::order::<usize>(n, a_p.raw_storage(), &a_i, &self.amd_control) {
            Err(status) => return Err(format!("amd error: {:?}", status)),
            Ok((p, _p_inv, _info)) => p,
        };

        let mut rhs: Vec<&mut [f64]> = vec![b.as_slice_mut().unwrap()];

        let lu = rlu::factor::<usize, f64>(
            n,
            &a_i,
            a_p.raw_storage(),
            &a,
            Some(&col_perm),
            &self.rlu_options,
        )?;

        rlu::solve(&lu, &mut rhs, false)
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
