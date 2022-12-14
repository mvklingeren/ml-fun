/* Ported from Python to Rust */
/* source: https://tutorialspoint.dev/language/python/creating-a-simple-machine-learning-model */

use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array2, Array1};
use rand::Rng;

fn main() {
    /* 1. Generate Training Set */
    let mut rng = rand::thread_rng();

    const TRAIN_SET_LIMIT:f32 = 1000.0;
    const TRAIN_SET_COUNT:usize = 100;

    let mut train_input= Array2::<f32>::zeros((100,3));
    let mut train_output = Array1::<f32>::zeros(100);

    for i in 1..TRAIN_SET_COUNT {
        let a = rng.gen_range(0.0..TRAIN_SET_LIMIT);
        let b = rng.gen_range(0.0..TRAIN_SET_LIMIT);
        let c = rng.gen_range(0.0..TRAIN_SET_LIMIT);
        let op = a + (2.0 * b) + (3.0 * c);

        train_input[[i,0]] = a;
        train_input[[i,1]] = b;
        train_input[[i,2]] = c;
        train_output[i] = op;
    }

    /* 2. Training the model */
    let dataset = DatasetBase::new(train_input, train_output);
    let model = LinearRegression::default().fit(&dataset).unwrap();
 
    /* 3. Testing our model */
    // Create our testing data set, the ouput should be 10*10 + 2*20 + 3*30 = 230
    // Random Test data
    let mut test_data = Array2::<f32>::zeros((1, 3));
    test_data[[0,0]]=10.0;
    test_data[[0,1]]=20.0;
    test_data[[0,2]]=30.0;

    //# Predict the ouput of the test data using the linear model
    let outcome = model.predict( test_data ); 
    println!("Outcome: {:?}",outcome.targets().first().unwrap().round());
    assert_eq!(outcome.targets().first().unwrap().round(), 140.0);
}
