use fast_math::exp;
use rand::Rng;
use raylib::prelude::*;

//INIZIO COSTANTI NN
//GESTIONE LEARNING
const NUM_TEST: u32 = 25 * 1000;
const EPS: f64 = 0.15;
const MAX_RATE: f64 = 5.0;

//GESIONE INPUT
const INPUT: usize = 784; // 28 * 28 * 3
const NUM_TRAIN_SAMPLE: usize = 1; // Numero di immagini in questo caso

//GESTIONE ARCHITETTURA
const LEN_ARC: usize = 3;
const ARC: [usize; LEN_ARC] = [2, 3, 2];
const NUM_NEURONS: usize = 7;

//FINE COSTANTI NN

//INIZIO COSTANTI PER RAYLIB
const SCREEN_HEIGHT: i32 = 720;
const SCREEN_WIDTH: i32 = (SCREEN_HEIGHT * 16) / 9;

//COSTANTI BASE
const TEXT_DIM: i32 = 23;
const DEF_PADDING: i32 = 30;
const DEF_MARGIN: i32 = 8;

//GESTIONE DELLA BARRA SUPERIORE
const TOP_BAR_POS: IntVec2 = IntVec2 { x: 75, y: 40 };
const TOP_BAR_DIM: IntVec2 = IntVec2 {
    x: SCREEN_WIDTH / 4 * 3,
    y: 0,
};

//GESTIONE DELLA BARRA DEL RATE
const RATE_BAR_H_PADDING: i32 = SCREEN_WIDTH / 16;
const RATE_BAR_POS: IntVec2 = IntVec2 {
    x: RATE_BAR_H_PADDING,
    y: SCREEN_HEIGHT / 9 + TOP_BAR_POS.y + TOP_BAR_DIM.y,
};
const RATE_BAR_DIM: IntVec2 = IntVec2 {
    x: SCREEN_WIDTH - 2 * RATE_BAR_H_PADDING,
    y: 0,
};
const RATE_BAR_RADIUS: f32 = 15.0;

//GESTIONE DEL GRAFICO
const GRAPHIC_POS: IntVec2 = IntVec2 {
    x: RATE_BAR_POS.x,
    y: RATE_BAR_POS.y + DEF_PADDING * 2,
};
const GRAPHIC_DIM: IntVec2 = IntVec2 { x: 600, y: 400 };

//GESTIONE DELLA VISUALIZZAZIONE DELLA NN
const VISUAL_NN_POS: IntVec2 = IntVec2 {
    x: SCREEN_WIDTH / 16 * 10,
    y: SCREEN_HEIGHT / 5 * 3,
};
const VISUAL_NN_DIM: IntVec2 = IntVec2 {
    x: SCREEN_WIDTH - DEF_PADDING - VISUAL_NN_POS.x,
    y: SCREEN_HEIGHT - DEF_PADDING - VISUAL_NN_POS.y,
};
const VISUAL_NN_NEURONS_RADIUS: f32 = 20.0;

//GESTIONE IMMAGINI
const IMG_1_FPATH: &str = "images/8.png";
const IMG_2_FPATH: &str = "images/6.png";
const IMG_POS: IntVec2 = IntVec2 {
    x: GRAPHIC_POS.x + GRAPHIC_DIM.x + DEF_PADDING * 4,
    y: RATE_BAR_POS.y + RATE_BAR_DIM.y + DEF_PADDING * 2,
};
const IMG_DIM: IntVec2 = IntVec2 { x: 100, y: 100 };

#[derive(Debug, Default)]
struct Neuron {
    w: Vec<f64>,
    b: f64,
}

#[derive(Debug, Default, Clone)]
struct IntVec2 {
    x: i32,
    y: i32,
}

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(SCREEN_WIDTH, SCREEN_HEIGHT)
        .title("Neural Network")
        .build();

    let mut nrs: [Neuron; NUM_NEURONS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let mut cont = 0;
    let mut cost: f64;
    let mut costs_vec: Vec<f64> = Vec::new();

    for layout_idx in 0..LEN_ARC {
        for _ in 0..ARC[layout_idx] {
            nrs[cont] = get_randomize_neuron(layout_idx);
            cont += 1;
        }
    }

    let mut rate: f64 = 1.2;

    let mut cont: u32 = 0;
    let mut max_cost: f64 = -1.0;

    let mut rate_circle: IntVec2 = IntVec2 {
        x: ((rate / MAX_RATE) * (SCREEN_WIDTH - 2 * RATE_BAR_H_PADDING) as f64) as i32
            + RATE_BAR_H_PADDING,
        y: RATE_BAR_POS.y,
    };

    let mut is_rate_button_clicked = false;

    //RESIZE AND LOAD IMAGES
    let mut img1 = Image::load_image(IMG_1_FPATH).unwrap();
    let color_vec_1 = nn_get_colors_array(img1.get_image_data());
    Image::resize(&mut img1, IMG_DIM.x, IMG_DIM.y);
    let texture_1 = rl.load_texture_from_image(&thread, &img1).unwrap();

    let mut img2 = Image::load_image(IMG_2_FPATH).unwrap();
    let color_vec_2 = nn_get_colors_array(img2.get_image_data());
    Image::resize(&mut img2, IMG_DIM.x, IMG_DIM.y);
    let texture_2 = rl.load_texture_from_image(&thread, &img2).unwrap();

    let data: [[f64; INPUT]; NUM_TRAIN_SAMPLE] = [color_vec_1];

    //TODO:
    // -> creare una funzione che da una color_map ti da un vettore con tutti i colori in ordine
    // -> creare la funzione che faccia l'opposto
    // -> salvare i due vettori in un array DATA che verrà passato a cost
    // -> fare dei test per verificare se è strettamente necessario avere un layer di neuroni pari
    //    al numero di pixel * canali.

    //START DRAWING RAYLIB
    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::BLACK);

        if cont < NUM_TEST {
            cost = update_weights(&mut nrs, data, rate);

            //println!("{}", cost);
            if cont == 0 {
                max_cost = cost;
                costs_vec.push(cost);
            } else {
                costs_vec.push((max_cost - cost) / max_cost);
            }
            cont += 1;
        }

        nn_draw_graph(&mut d, &mut costs_vec);
        //nn_draw_lg_text(&mut d, &mut nrs);
        nn_draw_rate_bar(
            &mut d,
            &mut rate_circle,
            &mut is_rate_button_clicked,
            &mut rate,
        );
        nn_draw_infos(&mut d, rate, cont);
        nn_draw_neurons(&mut d, &mut nrs);
        //println!("{:#?}", nrs);
        d.draw_texture(&texture_1, IMG_POS.x, IMG_POS.y, Color::WHITE);
        d.draw_texture(&texture_2, IMG_POS.x + IMG_DIM.x, IMG_POS.y, Color::WHITE);
    }
}

fn sigmuid(x: f64) -> f64 {
    1.0 / (1.0 + exp(-x as f32)) as f64
}

fn randomf() -> f64 {
    (rand::thread_rng().gen_range(-1000..=1000) as f64) / 1000.0 * 5.0
}

fn cost(nrs: &mut [Neuron; NUM_NEURONS], data: &[[f64; INPUT]; NUM_TRAIN_SAMPLE]) -> f64 {
    let mut out: f64 = 0.0;

    for sample_idx in 0..NUM_TRAIN_SAMPLE {
        //Per ogni tupla d'input
        let mut neuron_idx: usize = 0;
        let mut output: Vec<f64> = Vec::with_capacity(ARC[0]);

        for layer_idx in 0..LEN_ARC {
            //Per ogni Layer dell'architettura

            let input: Vec<f64>;
            input = output.clone();
            if layer_idx != 0 {
                output = Vec::with_capacity(ARC[layer_idx]);
            }

            for _ in 0..(ARC[layer_idx]) {
                //Per ogni Neurone del Layer
                let mut somma = 0.0;
                for inp_idx in 0..((nrs[neuron_idx].w).len()) {
                    //Per ogni input
                    //println!("{} {} {} {}", idx, d, nrs.len(), nrs[0].w.len());
                    if layer_idx == 0 {
                        somma += nrs[neuron_idx].w[inp_idx] * data[sample_idx][inp_idx];
                    } else {
                        somma += nrs[neuron_idx].w[inp_idx] * input[inp_idx];
                    }
                }
                somma += nrs[neuron_idx].b;
                output.push(sigmuid(somma));
                neuron_idx += 1;
            }
        }
        println!("{}", output.len());
        let mut tot_dst = 0.0;
        for i in 0..output.len() {
            tot_dst += output[i] - data[sample_idx][i];
        }

        tot_dst /= output.len() as f64;
        out += tot_dst * tot_dst;
    }

    //println!("FINITO");
    out / (NUM_TRAIN_SAMPLE as f64)
}

fn get_randomize_neuron(lidx: usize) -> Neuron {
    let mut n: Neuron = Neuron {
        w: Vec::new(),
        b: 0.0,
    };
    if lidx == 0 {
        n.w = Vec::with_capacity(INPUT);
        for _ in 0..INPUT {
            (n.w).push(randomf());
        }
    } else {
        n.w = Vec::with_capacity(ARC[(lidx - 1) as usize]);
        for _ in 0..ARC[(lidx - 1) as usize] {
            (n.w).push(randomf());
        }
    }

    n.b = randomf();
    n
}

fn update_weights(
    nrs: &mut [Neuron; NUM_NEURONS],
    data: [[f64; INPUT]; NUM_TRAIN_SAMPLE],
    rate: f64,
) -> f64 {
    //let cst: f64 = cost(nrs);
    let mut cst: f64 = 0.0;
    let mut idx: usize = 0;

    for layer_idx in 0..LEN_ARC {
        //Per ogni Layer dell'architettura
        for _ in 0..ARC[layer_idx] {
            cst = cost(nrs, &data);
            //Per ogni Neurone del Layer
            let weight: Vec<f64> = (nrs[idx].w).clone();
            let bias: f64 = nrs[idx].b;
            for inp_idx in 0..((nrs[idx].w).len()) {
                //Per ogni input

                nrs[idx].w[inp_idx] += EPS;
                let dw: f64 = (cost(nrs, &data) - cst) / EPS;
                nrs[idx].w[inp_idx] = weight[inp_idx];
                nrs[idx].w[inp_idx] -= dw * rate;
            }

            nrs[idx].b += EPS;
            let dw: f64 = (cost(nrs, &data) - cst) / EPS;
            nrs[idx].b = bias;
            nrs[idx].b -= dw * rate;
            idx += 1;
        }
    }

    cst
}

//stampa a terminale il risultato e lo ritorna anche
fn nn_get_result(nrs: &mut [Neuron; NUM_NEURONS], data: [f64; INPUT], print: bool) -> f64 {
    let mut idx: usize = 0;
    let mut res: f64 = -1.0;
    let mut output: Vec<f64> = Vec::with_capacity(ARC[0]);

    for b in 0..LEN_ARC {
        //Per ogni Layer dell'architettura
        /*
        In questa parte andiamo a copiare il risultato del output che
        diventerà l'input per la prossima serie di neuroni. In più se
        non siamo alla prima simulazione il vettore output verrà
        reinizializzato
        */
        let input: Vec<f64> = output.clone();
        if b != 0 {
            output = Vec::with_capacity(ARC[b]);
        }

        for _ in 0..(ARC[b]) {
            //Per ogni Neurone del Layer
            let mut somma = 0.0;
            for d in 0..((nrs[idx].w).len()) {
                //Per ogni input
                //println!("{} {} {} {} {}", idx, a, b, c, d);
                if b == 0 {
                    somma += nrs[idx].w[d] * data[d];
                } else {
                    somma += nrs[idx].w[d] * input[d];
                }
            }
            somma += nrs[idx].b;
            output.push(sigmuid(somma));
            idx += 1;
        }
        res = output[0];
    }

    if print {
        println!("{} {} {}", data[0], data[1], res);
    }

    res
}

fn nn_draw_graph(d: &mut RaylibDrawHandle<'_>, costs_vec: &mut Vec<f64>) {
    d.draw_line(
        GRAPHIC_POS.x - DEF_MARGIN,
        GRAPHIC_POS.y,
        GRAPHIC_POS.x - DEF_MARGIN,
        GRAPHIC_POS.y + GRAPHIC_DIM.y + DEF_MARGIN,
        Color::WHITE,
    );
    d.draw_line(
        GRAPHIC_POS.x - DEF_MARGIN,
        GRAPHIC_POS.y + GRAPHIC_DIM.y + DEF_MARGIN,
        GRAPHIC_POS.x + GRAPHIC_DIM.x,
        GRAPHIC_POS.y + GRAPHIC_DIM.y + DEF_MARGIN,
        Color::WHITE,
    );

    let vec_len = costs_vec.len();

    let mut x_prec: i32 = -1;
    let mut y_prec: i32 = -1;

    for n_point in 1..vec_len {
        let x =
            (((n_point as f64) / (vec_len as f64)) * GRAPHIC_DIM.x as f64) as i32 + GRAPHIC_POS.x;
        let y = (costs_vec[n_point] * GRAPHIC_DIM.y as f64) as i32 + GRAPHIC_POS.y;
        //Se i due cerchi hanno le stesse coordinate evito di disegnarli due volte
        if !((x_prec == x) && (y_prec == y)) {
            d.draw_circle(x, y, 2.0, Color::RED);
        }
        x_prec = x;
        y_prec = y;
    }
}

fn nn_draw_rate_bar(
    d: &mut RaylibDrawHandle<'_>,
    rate_circle: &mut IntVec2,
    is_rate_button_clicked: &mut bool,
    rate: &mut f64,
) {
    d.draw_line(
        RATE_BAR_POS.x,
        RATE_BAR_POS.y,
        RATE_BAR_POS.x + RATE_BAR_DIM.x,
        RATE_BAR_POS.y + RATE_BAR_DIM.y,
        Color::WHITE,
    );

    //Vediamo se il cursore è sul cerchio
    if (((d.get_mouse_x() > (rate_circle.x - RATE_BAR_RADIUS as i32)
        && d.get_mouse_x() < (rate_circle.x + RATE_BAR_RADIUS as i32))
        && d.get_mouse_y() > (rate_circle.y - RATE_BAR_RADIUS as i32)
        && d.get_mouse_y() < (rate_circle.y + RATE_BAR_RADIUS as i32))
        || (*is_rate_button_clicked))
        && (d.get_mouse_x() >= RATE_BAR_POS.x && d.get_mouse_x() <= RATE_BAR_POS.x + RATE_BAR_DIM.x)
    {
        *is_rate_button_clicked = true;
        if d.is_mouse_button_down(MouseButton::MOUSE_LEFT_BUTTON) {
            rate_circle.x = d.get_mouse_x();
            *rate =
                ((rate_circle.x - RATE_BAR_H_PADDING) as f64 / RATE_BAR_DIM.x as f64) * MAX_RATE;
        } else {
            *is_rate_button_clicked = false;
        }
    }

    d.draw_circle(rate_circle.x, rate_circle.y, RATE_BAR_RADIUS, Color::PINK);
}

fn nn_draw_infos(d: &mut RaylibDrawHandle<'_>, rate: f64, cont: u32) {
    d.draw_text(
        &format!("Rate: {:.2}", rate),
        TOP_BAR_POS.x,
        TOP_BAR_POS.y,
        TEXT_DIM,
        Color::WHITE,
    );

    d.draw_text(
        &(format!("{} / {}", cont, NUM_TEST)),
        TOP_BAR_POS.x + TOP_BAR_DIM.x,
        TOP_BAR_POS.y + TOP_BAR_DIM.y,
        TEXT_DIM,
        Color::WHITE,
    );
}

fn nn_draw_neurons(d: &mut RaylibDrawHandle<'_>, nrs: &mut [Neuron; NUM_NEURONS]) {
    let mut idx: usize = 0;
    let mut bias: f64;
    let mut weight: f64;
    let mut max_bias: f64 = 0.0;
    let mut max_w: f64 = 0.0;
    let mut g: f64;
    let mut r: f64;
    //let mut inp_centers: [IntVec2; INPUT] = Default::default();
    let mut centers: Vec<IntVec2> = Vec::with_capacity(ARC[0]);
    let mut prev_centers: Vec<IntVec2> = centers.clone();

    /*
    //Disegno i cerchi dell'input
    for l in 0..INPUT {
        let center: IntVec2 = IntVec2 {
            x: VISUAL_NN_POS.x + DEF_PADDING,
            y: VISUAL_NN_POS.y + (VISUAL_NN_DIM.y / (INPUT + 1) as i32) * (l + 1) as i32,
        };

        d.draw_circle(
            center.x,
            center.y,
            VISUAL_NN_NEURONS_RADIUS,
            Color {
                r: 0,
                g: 150,
                b: 150,
                a: 255,
            },
        );    data[0] = color_vec_1;
    data[1] = color_vec_2;


        inp_centers[l] = center;
    }
    */

    //Disegno la rete Neuroale
    for i in 0..LEN_ARC {
        //Per ogni neurone del layer
        for j in 0..ARC[i] {
            //per ogni bias
            bias = nrs[idx].b;

            if bias < 0.0 {
                g = 0.0;
                r = 255.0;
            } else {
                g = 255.0;
                r = 0.0;
            }

            bias = abs(bias);
            if bias > max_bias {
                max_bias = bias;
            }
            bias /= max_bias;

            let center: IntVec2 = IntVec2 {
                x: VISUAL_NN_POS.x
                    + DEF_PADDING
                    + (VISUAL_NN_DIM.x / (LEN_ARC + 1) as i32) * (i + 1) as i32,
                y: VISUAL_NN_POS.y + (VISUAL_NN_DIM.y / (ARC[i] + 1) as i32) * (j + 1) as i32,
            };

            //println!("{} {}", prev_centers.len(), centers.len());
            for x in 0..nrs[idx].w.len() {
                weight = (nrs[idx].w)[x];
                if weight < 0.0 {
                    g = 0.0;
                    r = 255.0;
                } else {
                    g = 255.0;
                    r = 0.0;
                }

                weight = abs(weight);
                if weight > max_w {
                    max_w = weight;
                }
                weight /= max_w;

                if i == 0 {
                    continue;
                    /*
                    d.draw_line(
                        inp_centers[x].x,
                        inp_centers[x].y,
                        center.x,
                        center.y,
                        Color {
                            r: (r * weight) as u8,
                            g: (g * weight) as u8,
                            b: 0,
                            a: 255,
                        },

                    );*/
                } else {
                    d.draw_line(
                        prev_centers[x].x,
                        prev_centers[x].y,
                        center.x,
                        center.y,
                        Color {
                            r: (r * weight) as u8,
                            g: (g * weight) as u8,
                            b: 0,
                            a: 255,
                        },
                    );
                }
            }

            d.draw_circle(
                center.x,
                center.y,
                VISUAL_NN_NEURONS_RADIUS,
                Color {
                    r: (r * bias) as u8,
                    g: (g * bias) as u8,
                    b: 0,
                    a: 255,
                },
            );

            centers.push(center);

            idx += 1;
        }

        prev_centers = centers.clone();
        centers = Vec::with_capacity(ARC[i]);
    }
}

fn abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

fn nn_get_colors_array(color_map: ImageColors) -> [f64; INPUT] {
    let mut out: [f64; INPUT] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let len: usize = color_map.len();
    for i in 0..len {
        out[i] = color_map[i].r as f64 / 255 as f64;
    }

    //println!("{:#?}", out);

    out
}
