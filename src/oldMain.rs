/*
In questo momento il codice è statico, bisogna renderlo dinamico
usando la memoria dinamica in modo da poter automaticamente allocare
un quantitativo corretto di memoria e non un valore predefinito.
*/
use rand::Rng;
use fast_math::exp;

const INPUT: usize = 2;
const NUM_TRAIN_SAMPLE: usize = 4;
const NUM_TEST: u32 = 100 * 1000;
const EPS: f64 = 0.1;
const RATE: f64 = 1.3;
const NUM_NEURONS: usize = 3;

const AND: [[f64; 3]; NUM_TRAIN_SAMPLE] = [
    [0.0,0.0,0.0],
    [0.0,1.0,0.0],
    [1.0,0.0,0.0],
    [1.0,1.0,1.0],
    ];

const OR: [[f64; 3]; NUM_TRAIN_SAMPLE] = [
        [0.0,0.0,0.0],
        [0.0,1.0,1.0],
        [1.0,0.0,1.0],
        [1.0,1.0,1.0],
    ];

const XOR: [[f64; 3]; NUM_TRAIN_SAMPLE] = [
        [0.0,0.0,0.0],
        [0.0,1.0,1.0],
        [1.0,0.0,1.0],
        [1.0,1.0,0.0],
    ];

const NAND: [[f64; 3]; NUM_TRAIN_SAMPLE] = [
        [0.0,0.0,1.0],
        [0.0,1.0,1.0],
        [1.0,0.0,1.0],
        [1.0,1.0,0.0],
    ];

const DATA: [[f64; 3]; NUM_TRAIN_SAMPLE] = XOR;
const LEN_ARC: usize = 2;
const ARC: [usize; LEN_ARC] = [2,1];

#[derive(Debug)]
#[derive(Default)]
struct Neuron {
    w: [f64; INPUT],
    b: f64,
}

fn main(){

    let mut nrs: [Neuron; NUM_NEURONS] = Default::default();
    for i in 0..NUM_NEURONS{
        nrs[i] = getRandomizedNeuron();
    }   

    for _ in 0..NUM_TEST {
            updateWeights(&mut nrs, 1);
        }   
     
    
    for i in 0..=1{
        for k in 0..=1{
            printResults(&mut nrs, [i as f64,k as f64]);
        }
    }

    println!("------------------------------------");

    let mut cont: i32 = 0;
    for n in nrs {
        println!("{} Neuron -----------------------------", cont);
        for i in 0..=1{
            for k in 0..=1{
                println!("{} {} {}", i, k, sigmuid(n.w[0]*(i as f64) + n.w[1] * (k as f64) + n.b));
            }
        }
        cont += 1;
        println!("------------------------------------");
    }
    
}

fn sigmuid(x: f64) -> f64 {    
    1.0 / (1.0 + exp(-x as f32)) as f64
}

fn randomf() -> f64 {
    (rand::thread_rng().gen_range(-1000..=1000) as f64) / 1000.0
}

fn cost(Neurons: &mut [Neuron; NUM_NEURONS]) -> f64 {
    let mut out: f64 = 0.0;
    let mut res: f64 = -1.0;

    for a in 0..NUM_TRAIN_SAMPLE{ //Per ogni tupla d'input

        let mut output: Vec<f64> = Vec::with_capacity(ARC[0]);
        let mut idx: usize = 0;

        for b in 0..LEN_ARC { //Per ogni Layer dell'architettura
            
            let mut input: Vec<f64> = Vec::with_capacity(output.len());
            input = output.clone();
            if b != 0 {
                output = Vec::with_capacity(ARC[b]);
            }

            for c in 0..(ARC[b]){ //Per ogni Neurone del Layer                
                
                let mut inp = 0.0;
                for d in 0..INPUT{ //Per ogni input
                    //println!("{} {} {} {} {}", idx, a, b, c, d);
                    if b == 0{
                        inp += Neurons[idx].w[d] * DATA[a][d];
                    }
                    else {
                        inp += Neurons[idx].w[d] * input[d];
                    }
                }
                
                inp += Neurons[idx].b;
                output.push(sigmuid(inp));
                idx += 1;
            }
            res = output[0];
        }
        let dst:    f64  = res - DATA[a][INPUT];         
        out += dst * dst;
    }

    out / (NUM_TRAIN_SAMPLE as f64) 

}

fn getRandomizedNeuron() -> Neuron {
    let mut n: Neuron = Neuron{ w: [0.0 , 0.0], b: 0.0};
    for i in 0..INPUT {
        n.w[i] = randomf(); 
    }
    n.b = randomf();
    n
}

fn updateWeights(nrs: &mut [Neuron; NUM_NEURONS], pr: u8) {
    let cst: f64 =  cost(nrs); 
    //println!("{}", cst);

    let mut idx: usize = 0;

    for a in 0..LEN_ARC { //Per ogni Layer dell'architettura
            for _ in 0..ARC[a]{ //Per ogni Neurone del Layer 
                let weight = nrs[idx].w;    
                let bias = nrs[idx].b;
                for c in 0..INPUT{ //Per ogni input
                    
                    nrs[idx].w[c] += EPS;
                    let dw: f64 = (cost(nrs) - cst) / EPS; 
                    nrs[idx].w[c] = weight[c];
                    nrs[idx].w[c] -= dw * RATE;
                }

                nrs[idx].b += EPS;
                let dw: f64 = (cost(nrs) - cst) / EPS; 
                nrs[idx].b = bias;
                nrs[idx].b -= dw * RATE;
                idx += 1;
            }
        }
    
    if pr == 1 {
        println!("-----------------------");
        for i in 0..=1{
            for k in 0..=1{
                printResults(nrs, [i as f64,k as f64]);
            }
        }
        println!("------------------------");
    }
}

fn printResults(nrs: &mut [Neuron; NUM_NEURONS], data: [f64; INPUT]){
    let mut idx: usize = 0;
    let mut y: [f64; INPUT] = [0.0, 0.0];
    
    for b in 0..LEN_ARC { //Per ogni Layer dell'architettura
        for c in 0..ARC[b]{ //Per ogni Neurone del Layer                
                let mut inp = 0.0;
                for d in 0..INPUT{ //Per ogni input
                    if b == 0{
                        inp += nrs[idx].w[d] * data[d];
                    }
                    else {
                        inp += nrs[idx].w[d] * y[d];
                    }
                }
                inp += nrs[idx].b;
                y[c] = sigmuid(inp);
                idx += 1;
            }
    }

    println!("{} {} {}", data[0], data[1], y[0])
}