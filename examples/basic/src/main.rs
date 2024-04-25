use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

fn main() {
    let model_options = ModelOptions {
        n_gpu_layers: 32,
        ..Default::default()
    };

    let llama = LLama::new(std::env::args().nth(1).unwrap(), &model_options).unwrap();

    let elapsed = std::time::Instant::now();

    let predict_options = PredictOptions {
        tokens: 100,
        top_k: 90,
        top_p: 0.86,
        ..Default::default()
    };

    println!(
        "Answer: {}",
        llama
            .predict(
                "What is the national animal of india?".into(),
                predict_options,
            )
            .unwrap()
    );

    println!("Elapsed: {:.2?}", elapsed.elapsed());
}
