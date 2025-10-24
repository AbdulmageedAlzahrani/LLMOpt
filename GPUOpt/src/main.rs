use std::env;
use std::process::Command;
use rusqlite::{Connection, Result};


struct Arg {
    name: String,
    value: String,
}

struct Config {
    executable_path: String,

}
fn main() {
    let args: Vec<String> = env::args().collect();
    let mut parsed_args = Vec::new();
    let output_report = String::from("output_report");

    for i in 1..args.len() {
        let arg = &args[i];
        if arg.starts_with("--") {
            let name = arg[2..].to_string();
            let value = args.get(i + 1).cloned().unwrap_or_default();
            parsed_args.push(Arg { name, value });
        }
    }

    let config = parse_config(&parsed_args);

    execute_nsys(&config, &output_report);

    analyze_report(format!("{}.sqlite", &output_report).as_str()).expect("Failed to analyze report");




}

fn parse_config(args: &[Arg]) -> Config {
    let mut executable_path = String::new();

    for arg in args {
        match arg.name.as_str() {
            "exec" => executable_path = arg.value.clone(),
            _ => {}
        }
    }

    Config { executable_path }
}

fn execute_nsys(config: &Config, output_report: &str) {
    let status = Command::new("nsys")
        .args([
            "profile",
            "--trace=cuda,cublas,cudnn,osrt,nvtx",
            "--stats=true",
            "-f", "true",
            "-o", output_report,
            config.executable_path.as_str(),
        ])
        .status()
        .expect("failed to run nsys");

    if !status.success() {
        eprintln!("nsys command failed with status: {}", status);
    }
}

fn analyze_report(db_path: &str) -> Result<()> {
    let conn = Connection::open(db_path)?;

    let mut stmt = conn.prepare(
        "SELECT name, total_time FROM cuda_kernels ORDER BY total_time DESC LIMIT 10",
    )?;
    let kernel_iter = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
    })?;

    println!("Top 10 CUDA Kernels by Total Time:");
    for kernel in kernel_iter {
        let (name, total_time) = kernel?;
        println!("Kernel: {}, Total Time: {} ms", name, total_time);
    }

    Ok(())
}


