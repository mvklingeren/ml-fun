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

    /* The ML Model -Linear Regression */
    let dataset = DatasetBase::new(train_input, train_output);
    let model = LinearRegression::default().fit(&dataset).unwrap();
 
    
    // Create our testing data set, the ouput should be 10*10 + 2*20 + 3*30 = 230
    let mut bla = Array2::<f32>::zeros((1, 3));
    bla[[0,0]]=10.0;
    bla[[0,1]]=20.0;
    bla[[0,2]]=30.0;
    let outcome = model.predict( bla ); //# Predict the ouput of the test data using the linear model
     
    println!("Outcome: {:?}",outcome);
}
