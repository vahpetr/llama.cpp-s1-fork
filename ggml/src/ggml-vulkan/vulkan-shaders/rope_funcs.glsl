
float rope_yarn_ramp(const float low, const float high, const uint i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

uint rope_a_coord(const uint i0, const uint i01, const uint i02, const uint i03, rope_params p) {
#if RMS_NORM_ROPE_FUSION
    // Per-row offset in shared memory
    const uint ix = i0;
#else
    const uint ix = p.a_offset + i03*p.nb03 + i02*p.nb02 + i01*p.nb01 + i0;
#endif
    return ix;
}

void rope_yarn(const float theta_extrap, const uint i0, out float cos_theta, out float sin_theta, rope_params p) {
    float mscale = p.attn_factor;
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = p.freq_scale * theta_extrap;
    float theta = theta_interp;
    if (p.ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(p.corr_dims[0], p.corr_dims[1], i0) * p.ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / p.freq_scale);
    }
    // Backprogagation uses inverted rotation
    if (p.is_back != 0) {
        theta = -theta;
    }
    cos_theta = cos(theta) * mscale;
    sin_theta = sin(theta) * mscale;
}

#if ROPE_PRECISE
// [TAG_ROPE_PRECISE] LLAMA_ROPE_PRECISE=1 variants: the angle in double. The fp32 path forms pos * pow(theta_scale, i) in fp32
// and hands |theta| up to 2^20 rad to the GPU's sin/cos, which scales it by an fp32 1/(2*pi) (-4.0e-8 relative) and rounds
// the revolutions to fp32: 0.09 rad of error at pos 1M. Here theta_scale is refined to double (two Newton steps on
// x^(n_dims/2) * freq_base = 1 from the fp32 value), theta = pos * theta_scale^i and the YaRN mix are double, theta is
// reduced to [-pi, pi] in double, and only that remainder goes through the fp32 sin/cos.
double rope_pow_d(double x, uint e) {
    double r = 1.0lf;
    while (e != 0u) {
        if ((e & 1u) != 0u) {
            r *= x;
        }
        x *= x;
        e >>= 1;
    }
    return r;
}

double rope_theta_scale_d(const rope_params p) {
    const uint   k  = p.n_dims / 2u;
    const double b  = double(p.freq_base);
    const double rk = double(1.0f / float(k)); // exact for a power-of-two k; a Newton step tolerates its rounding
    double x = double(p.theta_scale);
    for (int it = 0; it < 2; ++it) {
        const double g = rope_pow_d(x, k) * b - 1.0lf; // |g| ~ 1e-6, then ~1e-13
        x -= x * g * (1.0lf - g) * rk;              // x - x*g/(k*(1+g)) without an fp64 division
    }
    return x;
}

// ti = the pair's theta exponent (iw/2), i0 = its relative element index (for the YaRN ramp)
void rope_yarn_precise(const int pos, const uint ti, const float freq_factor, const uint i0, out float cos_theta, out float sin_theta, rope_params p) {
    float mscale = p.attn_factor;
    const double theta_extrap = double(pos) * rope_pow_d(rope_theta_scale_d(p), ti) / double(freq_factor);
    double theta = double(p.freq_scale) * theta_extrap;
    if (p.ext_factor != 0.0f) {
        const double y = (double(i0 / 2) - double(p.corr_dims[0])) / max(0.001lf, double(p.corr_dims[1]) - double(p.corr_dims[0]));
        const double ramp_mix = (1.0lf - min(1.0lf, max(0.0lf, y))) * double(p.ext_factor);
        theta = theta * (1.0lf - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / p.freq_scale);
    }
    if (p.is_back != 0) {
        theta = -theta;
    }
    double rev = theta * 0.15915494309189533576888376337251lf;
    rev -= roundEven(rev);
    const float a = float(rev) * 6.28318530717958647692f;
    cos_theta = cos(a) * mscale;
    sin_theta = sin(a) * mscale;
}
#endif

void rope_norm(const uint i0, const uint i1, const uint i2, const uint i3, rope_params p) {
    if (i0 >= p.ne00) {
        return;
    }

    uint idst = i0 + i1 * p.nb11 + i2 * p.nb12 + i3 * p.nb13;
    const uint ix = rope_a_coord(i0, i1, i2, i3, p);

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in rope_data_i.
    if (p.set_rows_stride != 0) {
        idst = i1*p.nb11 + i0;
        idst += rope_data_i[i2].x * p.set_rows_stride;
    }
    idst += p.d_offset;

    if (i0 < p.n_offs || i0 >= p.n_offs + p.n_dims) {
        rope_data_d[idst + 0] = ROPE_D_TYPE(rope_data_a[ix + 0]);
        rope_data_d[idst + 1] = ROPE_D_TYPE(rope_data_a[ix + 1]);

        return;
    }

    const uint iw = i0 - p.n_offs; // relative idx

#if ROPE_PRECISE
    const float freq_factor = p.has_ff != 0 ? rope_data_ff[iw/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn_precise(rope_data_pos[i2], iw/2, freq_factor, iw, cos_theta, sin_theta, p);
#else
    const float theta_base = rope_data_pos[i2] * pow(p.theta_scale, iw/2.0f);

    const float freq_factor = p.has_ff != 0 ? rope_data_ff[iw/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, iw, cos_theta, sin_theta, p);
#endif

    const float x0 = float(rope_data_a[ix + 0]);
    const float x1 = float(rope_data_a[ix + 1]);

    rope_data_d[idst + 0] = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
    rope_data_d[idst + 1] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
}

void rope_neox(const uint i0, const uint i1, const uint i2, const uint i3, rope_params p) {
    if (i0 >= p.ne00) {
        return;
    }

    uint idst = i0/2 + i1 * p.nb11 + i2 * p.nb12 + i3 * p.nb13;
    const uint ix = rope_a_coord(i0/2, i1, i2, i3, p);

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in rope_data_i.
    if (p.set_rows_stride != 0) {
        idst = i1*p.nb11 + i0/2;
        idst += rope_data_i[i2].x * p.set_rows_stride;
    }
    idst += p.d_offset;

    if (i0 < p.n_offs || i0 >= p.n_offs + p.n_dims) {
        rope_data_d[idst + i0/2 + 0] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 0]);
        rope_data_d[idst + i0/2 + 1] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 1]);

        return;
    }

    const uint iw = i0 - p.n_offs; // relative idx

#if ROPE_PRECISE
    const float freq_factor = p.has_ff != 0 ? rope_data_ff[iw/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn_precise(rope_data_pos[i2], iw/2, freq_factor, iw, cos_theta, sin_theta, p);
#else
    const float theta_base = rope_data_pos[i2] * pow(p.theta_scale, iw/2.0f);

    const float freq_factor = p.has_ff != 0 ? rope_data_ff[iw/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, iw, cos_theta, sin_theta, p);
#endif

    // idst/ix point at channel i0/2; the first channel of the rotated pair is p.n_offs + iw/2 = i0/2 + p.n_offs/2
    const float x0 = float(rope_data_a[ix + p.n_offs/2 + 0]);
    const float x1 = float(rope_data_a[ix + p.n_offs/2 + p.n_dims/2]);

    rope_data_d[idst + p.n_offs/2 + 0]          = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
    rope_data_d[idst + p.n_offs/2 + p.n_dims/2] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
}


void rope_multi(const uint i0, const uint i1, const uint i2, const uint i3, rope_params p) {
    if (i0 >= p.ne00) {
        return;
    }

    uint idst = i0/2 + i1 * p.nb11 + i2 * p.nb12 + i3 * p.nb13;
    const uint ix = rope_a_coord(i0/2, i1, i2, i3, p);

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in rope_data_i.
    if (p.set_rows_stride != 0) {
        idst = i1*p.nb11 + i0/2;
        idst += rope_data_i[i2].x * p.set_rows_stride;
    }
    idst += p.d_offset;

    if (i0 < p.n_offs || i0 >= p.n_offs + p.n_dims) {
        rope_data_d[idst + i0/2 + 0] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 0]);
        rope_data_d[idst + i0/2 + 1] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 1]);

        return;
    }

    const uint iw = i0 - p.n_offs; // relative idx

    const int sect_dims = p.sections[0] + p.sections[1] + p.sections[2] + p.sections[3];
    const int sec_w = p.sections[1] + p.sections[0];
    const uint sector = (iw / 2) % sect_dims;

#if ROPE_PRECISE
    int pos_sel;
    if (p.is_imrope != 0) {
        if (sector % 3 == 1 && sector < 3 * p.sections[1]) {
            pos_sel = rope_data_pos[i2 + p.ne02 * 1];
        } else if (sector % 3 == 2 && sector < 3 * p.sections[2]) {
            pos_sel = rope_data_pos[i2 + p.ne02 * 2];
        } else if (sector % 3 == 0 && sector < 3 * p.sections[0]) {
            pos_sel = rope_data_pos[i2];
        } else {
            pos_sel = rope_data_pos[i2 + p.ne02 * 3];
        }
    } else {
        if (sector < p.sections[0]) {
            pos_sel = rope_data_pos[i2];
        } else if (sector >= p.sections[0] && sector < sec_w) {
            pos_sel = rope_data_pos[i2 + p.ne02 * 1];
        } else if (sector >= sec_w && sector < sec_w + p.sections[2]) {
            pos_sel = rope_data_pos[i2 + p.ne02 * 2];
        } else {
            pos_sel = rope_data_pos[i2 + p.ne02 * 3];
        }
    }

    const float freq_factor = p.has_ff != 0 ? rope_data_ff[iw/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn_precise(pos_sel, iw/2, freq_factor, iw, cos_theta, sin_theta, p);
#else
    float theta_base = 0.0;
    if (p.is_imrope != 0) {
        if (sector % 3 == 1 && sector < 3 * p.sections[1]) {
            theta_base = rope_data_pos[i2 + p.ne02 * 1]*pow(p.theta_scale, iw/2.0f);
        } else if (sector % 3 == 2 && sector < 3 * p.sections[2]) {
            theta_base = rope_data_pos[i2 + p.ne02 * 2]*pow(p.theta_scale, iw/2.0f);
        } else if (sector % 3 == 0 && sector < 3 * p.sections[0]) {
            theta_base = rope_data_pos[i2]*pow(p.theta_scale, iw/2.0f);
        } else {
            theta_base = rope_data_pos[i2 + p.ne02 * 3]*pow(p.theta_scale, iw/2.0f);
        }
    } else {
        if (sector < p.sections[0]) {
            theta_base = rope_data_pos[i2]*pow(p.theta_scale, iw/2.0f);
        }
        else if (sector >= p.sections[0] && sector < sec_w) {
            theta_base = rope_data_pos[i2 + p.ne02 * 1]*pow(p.theta_scale, iw/2.0f);
        }
        else if (sector >= sec_w && sector < sec_w + p.sections[2]) {
            theta_base = rope_data_pos[i2 + p.ne02 * 2]*pow(p.theta_scale, iw/2.0f);
        }
        else if (sector >= sec_w + p.sections[2]) {
            theta_base = rope_data_pos[i2 + p.ne02 * 3]*pow(p.theta_scale, iw/2.0f);
        }
    }

    const float freq_factor = p.has_ff != 0 ? rope_data_ff[iw/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, iw, cos_theta, sin_theta, p);
#endif

    // idst/ix point at channel i0/2; the first channel of the rotated pair is p.n_offs + iw/2 = i0/2 + p.n_offs/2
    const float x0 = float(rope_data_a[ix + p.n_offs/2 + 0]);
    const float x1 = float(rope_data_a[ix + p.n_offs/2 + p.n_dims/2]);

    rope_data_d[idst + p.n_offs/2 + 0]          = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
    rope_data_d[idst + p.n_offs/2 + p.n_dims/2] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
}

void rope_vision(const uint i0, const uint i1, const uint i2, const uint i3, rope_params p) {
    if (i0 >= p.ne00) {
        return;
    }

    const uint idst = p.d_offset + i0/2 + i1 * p.nb11 + i2 * p.nb12 + i3 * p.nb13;
    const uint ix = rope_a_coord(i0/2, i1, i2, i3, p);

    const int sect_dims = p.sections[0] + p.sections[1];
    const int sec_w = p.sections[1] + p.sections[0];
    const uint sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < p.sections[0]) {
        const uint p0 = sector;
        theta_base = rope_data_pos[i2]*pow(p.theta_scale, p0);
    }
    else if (sector >= p.sections[0] && sector < sec_w) {
        const uint p0 = sector - p.sections[0];
        theta_base = rope_data_pos[i2 + p.ne02]*pow(p.theta_scale, p0);
    }

    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0/2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);

    const float x0 = float(rope_data_a[ix + 0]);
    const float x1 = float(rope_data_a[ix + p.n_dims]);

    rope_data_d[idst + 0]        = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
    rope_data_d[idst + p.n_dims] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
}

