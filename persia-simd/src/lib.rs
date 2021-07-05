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
