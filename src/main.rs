use linfa::prelude::*;
use linfa_linear::LinearRegression;
use rand::Rng;

fn main() {
    /* Generate Training Set */
    let mut rng = rand::thread_rng();

    let train_set_limit = 1000;
    let train_set_count = 100;

    let mut train_input: Vec<[i32; 3]> = Vec::new();
    let mut train_output: Vec<i32> = Vec::new();
    for _x in 1..train_set_count {
        let a: i32 = rng.gen_range(0..train_set_limit); //randint(0, TRAIN_SET_LIMIT)
        let b = rng.gen_range(0..train_set_limit); //randint(0, TRAIN_SET_LIMIT)
        let c = rng.gen_range(0..train_set_limit); //randint(0, TRAIN_SET_LIMIT)
        let op = a + (2 * b) + (3 * c);

        train_input.push([a, b, c]);
        train_output.push(op);
    }

    /* The ML Model -Linear Regression */
    let predictor = LinearRegression::default().fit();
    //let predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)
}
