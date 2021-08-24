use std::arch::x86_64::*;

#[allow(clippy::missing_safety_doc)]
pub unsafe fn add_assign_avx2(a: &mut [f32], b: &[f32]) {
    let length = a.len();
    let end = (length as isize / 8) * 8;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    for i in (0..end).step_by(8) {
        let a_v = _mm256_loadu_ps(a_ptr.offset(i));
        let b_v = _mm256_loadu_ps(b_ptr.offset(i));
        let result_v = _mm256_add_ps(a_v, b_v);
        _mm256_storeu_ps(a_ptr.offset(i) as *mut f32, result_v);
    }
    for i in end as usize..length {
        a[i] += b[i];
    }
}

#[allow(clippy::missing_safety_doc)]
pub unsafe fn decayed_adagrad_avx2(
    adagrad: &mut [f32],
    embedding: &mut [f32],
    gradient: &[f32],
    g_square_momentum: f32,
    learning_rate: f32,
    eps: f32,
) {
    let length = adagrad.len();
    let end = (length / 8) * 8;
    let adagrad_ptr = adagrad.as_ptr();
    let embedding_ptr = embedding.as_ptr();
    let gradient_ptr = gradient.as_ptr();
    for i in (0..end as isize).step_by(8) {
        let adagrad_v = _mm256_loadu_ps(adagrad_ptr.offset(i));
        let embedding_v = _mm256_loadu_ps(embedding_ptr.offset(i));
        let gradient_v = _mm256_loadu_ps(gradient_ptr.offset(i));

        let squared_v = _mm256_mul_ps(gradient_v, gradient_v);

        let scaled_gradient_v = _mm256_mul_ps(
            gradient_v,
            _mm256_rsqrt_ps(_mm256_add_ps(adagrad_v, _mm256_set1_ps(eps))),
        );

        let embedding_result_v = _mm256_fnmadd_ps(
            _mm256_set1_ps(learning_rate),
            scaled_gradient_v,
            embedding_v,
        );

        _mm256_storeu_ps(embedding_ptr.offset(i) as *mut f32, embedding_result_v);

        let updated_adagrad_v =
            _mm256_fmadd_ps(adagrad_v, _mm256_set1_ps(g_square_momentum), squared_v);

        _mm256_storeu_ps(adagrad_ptr.offset(i) as *mut f32, updated_adagrad_v);
    }

    for i in end..length {
        let adagrad_v = &adagrad[i];
        let embedding_v = &embedding[i];
        let gradient_v = &gradient[i];

        let squared_v = gradient_v * gradient_v;

        let scaled_gradient_v = gradient_v * (adagrad_v + eps).sqrt().recip();

        let embedding_result_v = -learning_rate * scaled_gradient_v + embedding_v;

        embedding[i] = embedding_result_v;

        let updated_adagrad_v = adagrad_v * g_square_momentum + squared_v;

        adagrad[i] = updated_adagrad_v;
    }
}

#[allow(clippy::missing_safety_doc)]
pub unsafe fn decayed_sgd_avx2(emb: &mut [f32], grad: &[f32], wd: f32, lr: f32) {
    let length = emb.len();
    let end = (length / 8) * 8; // divide by simd step
    let grad_ptr = grad.as_ptr();
    let emb_ptr = emb.as_ptr();

    for i in (0..end as isize).step_by(8) {
        let grad_v = _mm256_loadu_ps(grad_ptr.offset(i));
        let emb_v = _mm256_loadu_ps(emb_ptr.offset(i));

        // weight decay
        let decay_grad_v = _mm256_fmadd_ps(_mm256_set1_ps(wd), emb_v, grad_v);
        let updated_emb = _mm256_fnmadd_ps(_mm256_set1_ps(lr), decay_grad_v, emb_v);
        _mm256_storeu_ps(emb_ptr.offset(i) as *mut f32, updated_emb);
    }

    for i in end..length {
        let decay_grad_v = grad[i] + emb[i] * wd;
        emb[i] = emb[i] - lr * decay_grad_v;
    }
}

#[allow(clippy::missing_safety_doc)]
pub unsafe fn adam_avx2(
    adam_m: &mut [f32],
    adam_v: &mut [f32],
    beta1_power: &[f32],
    beta2_power: &[f32],
    emb: &mut [f32],
    grad: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
) {
    let length = emb.len();
    let end = (length / 8) * 8; // divide by simd step
    let adam_m_ptr = adam_m.as_ptr();
    let adam_v_ptr = adam_v.as_ptr();
    let beta1_power_ptr = beta1_power.as_ptr();
    let beta2_power_ptr = beta2_power.as_ptr();
    let grad_ptr = grad.as_ptr();
    let emb_ptr = emb.as_ptr();

    for i in (0..end as isize).step_by(8) {
        let grad_vec = _mm256_loadu_ps(grad_ptr.offset(i));
        let emb_vec = _mm256_loadu_ps(emb_ptr.offset(i));
        let adam_v_vec = _mm256_loadu_ps(adam_v_ptr.offset(i));
        let adam_m_vec = _mm256_loadu_ps(adam_m_ptr.offset(i));
        let beta1_power_vec = _mm256_loadu_ps(beta1_power_ptr.offset(i));
        let beta2_power_vec = _mm256_loadu_ps(beta2_power_ptr.offset(i));

        let updated_m = _mm256_fmadd_ps(
            _mm256_set1_ps(beta1),
            adam_m_vec,
            _mm256_mul_ps(_mm256_set1_ps(1.0_f32 - beta1), grad_vec),
        );

        let updated_v = _mm256_fmadd_ps(
            _mm256_set1_ps(beta2),
            adam_v_vec,
            _mm256_mul_ps(
                _mm256_set1_ps(1.0 - beta2),
                _mm256_mul_ps(grad_vec, grad_vec),
            ),
        );

        let m_bias_corr = _mm256_div_ps(
            updated_m,
            _mm256_sub_ps(_mm256_set1_ps(1.0_f32), beta1_power_vec),
        );
        let v_bias_corr = _mm256_div_ps(
            updated_v,
            _mm256_sub_ps(_mm256_set1_ps(1.0_f32), beta2_power_vec),
        );

        let descent = _mm256_div_ps(
            m_bias_corr,
            _mm256_add_ps(_mm256_set1_ps(eps), _mm256_sqrt_ps(v_bias_corr)),
        );

        let updated_emb = _mm256_fnmadd_ps(_mm256_set1_ps(lr), descent, emb_vec);

        _mm256_storeu_ps(adam_m_ptr.offset(i) as *mut f32, updated_m);
        _mm256_storeu_ps(adam_v_ptr.offset(i) as *mut f32, updated_v);
        _mm256_storeu_ps(emb_ptr.offset(i) as *mut f32, updated_emb);
    }

    for i in end..length {
        let grad_val = &grad[i];
        let emb_val = &emb[i];
        let adam_v_val = &adam_v[i];
        let adam_m_val = &adam_m[i];
        let beta1_power_val = &beta1_power[i];
        let beta2_power_val = &beta2_power[i];

        let updated_m = beta1 * adam_m_val + (1.0_f32 - beta1) * grad_val;
        let updated_v = beta2 * adam_v_val + (1.0_f32 - beta2) * grad_val * grad_val;

        let m_bias_corr = updated_m / (1.0_f32 - beta1_power_val);
        let v_bias_corr = updated_v / (1.0_f32 - beta2_power_val);

        let descent = m_bias_corr / (eps + v_bias_corr.sqrt());

        let updated_emb = emb_val - lr * descent;

        adam_m[i] = updated_m;
        adam_v[i] = updated_v;
        emb[i] = updated_emb;
    }
}

#[allow(clippy::missing_safety_doc)]
pub unsafe fn weight_bound(embedding: &mut [f32], weight_bound: f32) {
    let length = embedding.len();
    let end = (length / 8) * 8;
    let embedding_ptr = embedding.as_ptr();
    let neg_weight_bound = -weight_bound;
    for i in (0..end as isize).step_by(8) {
        let embedding_v = _mm256_loadu_ps(embedding_ptr.offset(i));

        let bounded_embedding_v = _mm256_min_ps(
            _mm256_max_ps(embedding_v, _mm256_set1_ps(neg_weight_bound)),
            _mm256_set1_ps(weight_bound),
        );

        _mm256_storeu_ps(embedding_ptr.offset(i) as *mut f32, bounded_embedding_v);
    }

    for embedding_v in embedding.iter_mut().skip(end) {
        let bounded_embedding_v = embedding_v.max(neg_weight_bound).min(weight_bound);
        *embedding_v = bounded_embedding_v;
    }
}
