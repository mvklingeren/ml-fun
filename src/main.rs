use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array2, Array1, ArrayBase, Dim};
use rand::Rng;

fn main() {
    /* Generate Training Set */
    let mut rng = rand::thread_rng();

    let train_set_limit:f32 = 1000.0;
    let train_set_count = 100;

    let mut train_input= Array2::<f32>::zeros((100,3));
    let mut train_output = Array1::<f32>::zeros(100);

    for i in 1..train_set_count {
        let a: f32 = rng.gen_range(0.0..train_set_limit);
        let b = rng.gen_range(0.0..train_set_limit);
        let c = rng.gen_range(0.0..train_set_limit);
        let op = a + (2.0 * b) + (3.0 * c);

        train_input[[i,0]] = a;
        train_input[[i,1]] = b;
        train_input[[i,2]] = c;

        train_output[i] = op;
    }

    // let records = array![[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]];
    // let targets = array![1., 2., 3., 4., 5.];

    /* The ML Model -Linear Regression */
    let dataset = DatasetBase::new(train_input, train_output);
    let model = LinearRegression::default().fit(&dataset).unwrap();
 
    
    //let predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)
}
