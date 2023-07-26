//__CONSTANT__ float PI = 3.141592653589793f;

// DaVinci Intermediate Constants
__CONSTANT__ float DI_A = 0.0075f;
__CONSTANT__ float DI_B = 7.0f;
__CONSTANT__ float DI_C = 0.07329248f;
__CONSTANT__ float DI_M = 10.44426855f;
__CONSTANT__ float DI_LIN_CUT = 0.00262409f;
__CONSTANT__ float DI_LOG_CUT = 0.02740668f;

__DEVICE__ float3 DaVinci_Intermediate_to_linear(float3 rgb)
{
	rgb.x = (rgb.x > DI_LOG_CUT) ? _powf(2.0f, (rgb.x / DI_C) - DI_B)-DI_A : rgb.x/DI_M;
	rgb.y = (rgb.y > DI_LOG_CUT) ? _powf(2.0f, (rgb.y / DI_C) - DI_B)-DI_A : rgb.y/DI_M;
	rgb.z = (rgb.z > DI_LOG_CUT) ? _powf(2.0f, (rgb.z / DI_C) - DI_B)-DI_A : rgb.z/DI_M;
	
	return rgb;
}

__DEVICE__ float3 linear_to_DaVinci_Intermediate(float3 rgb)
{
	rgb.x = (rgb.x > DI_LIN_CUT) ? (_log2f(rgb.x + DI_A) + DI_B)*DI_C : rgb.x*DI_M;
	rgb.y = (rgb.y > DI_LIN_CUT) ? (_log2f(rgb.y + DI_A) + DI_B)*DI_C : rgb.y*DI_M;
	rgb.z = (rgb.z > DI_LIN_CUT) ? (_log2f(rgb.z + DI_A) + DI_B)*DI_C : rgb.z*DI_M;
	
	return rgb;
}

__DEVICE__ float3 DWG_to_XYZ(float3 rgb)
{
	float3 xyz;
	
	xyz.x = 0.70062239 * rgb.x + 0.14877482 * rgb.y + 0.10105872 * rgb.z;
	xyz.y = 0.27411851 * rgb.x + 0.87363190 * rgb.y + -0.14775041 * rgb.z;
	xyz.z = -0.09896291 * rgb.x + -0.13789533 * rgb.y + 1.32591599 * rgb.z;
	
	return xyz;
}

__DEVICE__ float3 XYZ_to_DWG(float3 xyz)
{
	float3 rgb;
	
	rgb.x =  1.51667204 * xyz.x + -0.28147805 * xyz.y + -0.14696363 * xyz.z;
	rgb.y = -0.46491710 * xyz.x +  1.25142378 * xyz.y +  0.17488461 * xyz.z;
	rgb.z =  0.06484905 * xyz.x +  0.10913934 * xyz.y +  0.76141462 * xyz.z;
	
	return rgb;
}

__DEVICE__ float3 DWG_to_Oklab_LCh(float3 rgb)
{
	float3 xyz = DWG_to_XYZ(rgb);
	float3 lms, lms_;
	
	lms.x = 0.8189330101 * xyz.x + 0.3618667424 * xyz.y + -0.1288597137 * xyz.z;
	lms.y = 0.0329845436 * xyz.x + 0.9293118715 * xyz.y + 0.0361456387 * xyz.z;
	lms.z = 0.0482003018 * xyz.x + 0.2643662691 * xyz.y + 0.6338517070 * xyz.z;
	
	lms_.x = spow(lms.x, 1.0f/3);
	lms_.y = spow(lms.y, 1.0f/3);
	lms_.z = spow(lms.z, 1.0f/3);
	
	float L = 0.2104542553 * lms_.x + 0.7936177850 * lms_.y + -0.0040720468 * lms_.z;
	float a = 1.9779984951 * lms_.x + -2.4285922050 * lms_.y + 0.4505937099 * lms_.z;
	float b = 0.0259040371 * lms_.x + 0.7827717662 * lms_.y + -0.8086757660 * lms_.z;
	
	// Lab to LCh
	float C = _sqrtf(a*a + b*b);
	float h = _atan2f(b, a);
	
	return make_float3(L, C, h);
}

__DEVICE__ float3 Oklab_LCh_to_DWG(float3 LCh)
{
	float3 Lab, lms, lms_, xyz;
	
	// LCh to Lab
	Lab.x = LCh.x;
	Lab.y = LCh.y * _cosf(LCh.z);
	Lab.z = LCh.y * _sinf(LCh.z);
	
	lms_.x = 0.9999999985 * Lab.x + 0.3963377922 * Lab.y + 0.2158037581 * Lab.z;
	lms_.y = 1.0000000089 * Lab.x + -0.1055613423 * Lab.y + -0.0638541748 * Lab.z;
	lms_.z = 1.0000000547 * Lab.x + -0.0894841821 * Lab.y + -1.2914855379 * Lab.z;
	
	lms.x = spow(lms_.x, 3);
	lms.y = spow(lms_.y, 3);
	lms.z = spow(lms_.z, 3);
	
	xyz.x = 1.2270138511 * lms.x + -0.5577999807 * lms.y + 0.2812561490 * lms.z;
	xyz.y = -0.0405801784 * lms.x + 1.1122568696 * lms.y + -0.0716766787 * lms.z;
	xyz.z = -0.0763812845 * lms.x + -0.4214819784 * lms.y + 1.5861632204 * lms.z;
	
	return XYZ_to_DWG(xyz);
}

__DEVICE__ float ZCAM_Iz_to_luminance(float Iz)
{
	float c1 = 3424 / _exp2f(12);
	float c2 = 2413 / _exp2f(7);
	float c3 = 2392 / _exp2f(7);
	float rho = 1.7 * 2523 / _exp2f(5);
	float zcam_luminance_shift = 1.0f / (-0.20151000f + 1.12064900f + 0.05310080f);
	float st2084_m_1=2610.0f / 4096.0f * (1.0f / 4.0f);
	float st2084_m_1_d = 1.0f / st2084_m_1;
	float st2084_L_p = 10000.0f;
	
	float V_p = _powf(Iz, 1 / rho);
    float luminance = _powf((_fmaxf(0, V_p - c1) / (c2 - c3 * V_p)), st2084_m_1_d)*st2084_L_p * zcam_luminance_shift;
    return luminance;
}

__DEVICE__ float luminance_to_ZCAM_Iz(float luminance)
{
	float c1 = 3424 / _exp2f(12);
	float c2 = 2413 / _exp2f(7);
	float c3 = 2392 / _exp2f(7);
	float rho = 1.7 * 2523 / _exp2f(5);
	float zcam_luminance_shift = 1.0f / (-0.20151000f + 1.12064900f + 0.05310080f);
	float st2084_m_1=2610.0f / 4096.0f * (1.0f / 4.0f);
	float st2084_m_1_d = 1.0f / st2084_m_1;
	float st2084_L_p = 10000.0f;
	
	float Y_p = _powf((luminance/zcam_luminance_shift) / st2084_L_p, st2084_m_1);
    float Iz = _powf((c1 + c2 * Y_p) / (c3 * Y_p + 1.0f), rho);
    return Iz;
}

__DEVICE__ float3 DWG_to_ZCAM_Izazbz(float3 rgb)
{
	// Step 0
	float3 xyz = DWG_to_XYZ(rgb);
	
	// Step 2
	float b = 1.15;
	float g = 0.66;
	float c1 = 3424 / _exp2f(12);
	float c2 = 2413 / _exp2f(7);
	float c3 = 2392 / _exp2f(7);
	float eta = 2610 / _exp2f(14);
	float rho = 1.7 * 2523 / _exp2f(5);
	float epsilon = 3.7035226210190005 * _powf(10, -11);
	
	float3 lms, lms_;
	
	float X_ = b * xyz.x - (b - 1) * xyz.z;
	float Y_ = g * xyz.y - (g - 1) * xyz.x;

	lms.x = 0.41478972 * X_ + 0.579999 * Y_ + 0.0146480 * xyz.z;
	lms.y = -0.2015100 * X_ + 1.120649 * Y_ + 0.0531008 * xyz.z;
	lms.z = -0.0166008 * X_ + 0.264800 * Y_ + 0.6684799 * xyz.z;
	
	// Reduce saturation of noise before the non-linearity so that there is just no clipping
	if (lms.y <= 0) lms.x = lms.z = 0;
	if (lms.x < 0 || lms.z < 0)
	{
		float min = _fminf(lms.x, lms.z);
		float max = _fmaxf(lms.x, lms.z);
		float L = lms.y;
		
		float min_new = 0;
		float max_new = L / (L - min) * (max - L) + L;
		
		if (lms.x == min) lms.x = min_new;
		else if (lms.z == min) lms.z = min_new;
		
		if (lms.x == max) lms.x = max_new;
		else if (lms.z == max) lms.z = max_new;
	}
	
	lms_.x = spow((c1 + c2 * spow(lms.x/10000, eta)) / (1 + c3 * spow(lms.x/10000, eta)), rho);
	lms_.y = spow((c1 + c2 * spow(lms.y/10000, eta)) / (1 + c3 * spow(lms.y/10000, eta)), rho);
	lms_.z = spow((c1 + c2 * spow(lms.z/10000, eta)) / (1 + c3 * spow(lms.z/10000, eta)), rho);
	
	float az = 3.524000 * lms_.x + -4.066708 * lms_.y + 0.542708 * lms_.z;
	float bz = 0.199076 * lms_.x + 1.096799 * lms_.y + -1.295875 * lms_.z;
	
	float Iz = lms_.y - epsilon;
	
	return make_float3(Iz, az, bz);
}

__DEVICE__ float3 ZCAM_Izazbz_to_DWG(float3 Izazbz)
{	
	float b = 1.15;
	float g = 0.66;
	float c1 = 3424 / _exp2f(12);
	float c2 = 2413 / _exp2f(7);
	float c3 = 2392 / _exp2f(7);
	float eta = 2610 / _exp2f(14);
	float rho = 1.7 * 2523 / _exp2f(5);
	float epsilon = 3.7035226210190005 * _powf(10, -11);
	
	float Iz = Izazbz.x;
	float az = Izazbz.y;
	float bz = Izazbz.z;
	
	// Step 5
	float3 lms, lms_, xyz, xyz_;
	
	float I = Iz + epsilon;
	
	lms_.x = 1.00000000 * I + 0.27721009 * az + 0.11609463 * bz;
	lms_.y = 1.00000000 * I;
	lms_.z = 1.00000000 * I + 0.04258580 * az + -0.75384458 * bz;
	
	lms.x = 10000 * spow((c1 - spow(lms_.x, 1/rho)) / (c3 * spow(lms_.x, 1/rho) - c2), 1/eta);
	lms.y = 10000 * spow((c1 - spow(lms_.y, 1/rho)) / (c3 * spow(lms_.y, 1/rho) - c2), 1/eta);
	lms.z = 10000 * spow((c1 - spow(lms_.z, 1/rho)) / (c3 * spow(lms_.z, 1/rho) - c2), 1/eta);
	
	xyz_.x = 1.92422644 *lms.x + -1.00479231 *lms.y + 0.03765140 *lms.z;
	xyz_.y = 0.35031676 *lms.x + 0.72648119 *lms.y + -0.06538442 *lms.z;
	xyz_.z = -0.09098281 *lms.x + -0.31272829 *lms.y + 1.52276656 *lms.z;
	
	xyz.x = (xyz_.x + (b - 1) * xyz_.z) / b;
	xyz.y = (xyz_.y + (g - 1) * xyz.x) / g;
	xyz.z = xyz_.z;
	
	return XYZ_to_DWG(xyz);
}

__DEVICE__ float3 ZCAM_Izazbz_to_ZCAM_JMh(float3 Izazbz)
{
	// Step 1
	float F_s = 0.59; // 0.525 = dark, 0.59 = dim, 0.69 = average
	float backgroundLuminance = 20;
	float referenceLuminance = 100;
	float F_b = _sqrtf(backgroundLuminance / referenceLuminance);
	float L_a = referenceLuminance * backgroundLuminance / 100;
	float F_L = 0.171 * _powf(L_a, 1/3) * (1 - _expf(-48/9*L_a));
	
	// Step 3
	float hz = _fmod(_atan2f(Izazbz.z, Izazbz.y) * 180/PI + 360, 360);
	
	float ez = 1.015 + _cosf((89.038 + hz) * PI/180);
	
	float Izw = DWG_to_ZCAM_Izazbz(make_float3(referenceLuminance, referenceLuminance, referenceLuminance)).x;
	
	float Qz = 2700 * _powf(Izazbz.x, 1.6*F_s/_powf(F_b, 0.12)) * _powf(F_s, 2.2) * _powf(F_b, 0.5) * _powf(F_L, 0.2);
	float Qzw = 2700 * _powf(Izw, 1.6*F_s/_powf(F_b, 0.12)) * _powf(F_s, 2.2) * _powf(F_b, 0.5) * _powf(F_L, 0.2);
	
	float Jz = 100 * Qz / Qzw;
	float Mz = 100 * _powf(Izazbz.y*Izazbz.y + Izazbz.z*Izazbz.z, 0.37) * (_powf(ez, 0.068) * _powf(F_L, 0.2)) / (_powf(F_b, 0.1) * _powf(Izw, 0.78));
	
	return make_float3(Jz, Mz, hz);
}

__DEVICE__ float3 DWG_to_ZCAM_JMh(float3 rgb)
{
	// Step 1
	float F_s = 0.59; // 0.525 = dark, 0.59 = dim, 0.69 = average
	float backgroundLuminance = 20;
	float referenceLuminance = 100;
	float F_b = _sqrtf(backgroundLuminance / referenceLuminance);
	float L_a = referenceLuminance * backgroundLuminance / 100;
	float F_L = 0.171 * _powf(L_a, 1/3) * (1 - _expf(-48/9*L_a));
	
	// Step 2
	float3 Izazbz = DWG_to_ZCAM_Izazbz(rgb);
	
	float Iz = Izazbz.x;
	float az = Izazbz.y;
	float bz = Izazbz.z;
	
	// Step 3
	float hz = _fmod(_atan2f(bz, az) * 180/PI + 360, 360);
	
	// Step 4
	float ez = 1.015 + _cosf((89.038 + hz) * PI/180);
	
	// Step 5
	//float3 XYZw = DWG_to_XYZ(make_float3(referenceLuminance, referenceLuminance, referenceLuminance));	// = 95.04559300, 100.00000000, 108.90577500
	float Izw = DWG_to_ZCAM_Izazbz(make_float3(referenceLuminance, referenceLuminance, referenceLuminance)).x;
	
	float Qz = 2700 * spow(Iz, 1.6*F_s/_powf(F_b, 0.12)) * _powf(F_s, 2.2) * _powf(F_b, 0.5) * _powf(F_L, 0.2);
	float Qzw = 2700 * spow(Izw, 1.6*F_s/_powf(F_b, 0.12)) * _powf(F_s, 2.2) * _powf(F_b, 0.5) * _powf(F_L, 0.2);
	
	float Jz = 100 * Qz / Qzw;
	float Mz = 100 * spow(az*az + bz*bz, 0.37) * (spow(ez, 0.068) * _powf(F_L, 0.2)) / (_powf(F_b, 0.1) * _powf(Izw, 0.78));
	
	return make_float3(Jz, Mz, hz);
}

__DEVICE__ float3 ZCAM_JMh_to_DWG(float3 JzMzhz)
{
	float F_s = 0.59; // 0.525 = dark, 0.59 = dim, 0.69 = average
	float backgroundLuminance = 20;
	float referenceLuminance = 100;
	float F_b = _sqrtf(backgroundLuminance / referenceLuminance);
	float L_a = referenceLuminance * backgroundLuminance / 100;
	float F_L = 0.171 * _powf(L_a, 1/3) * (1 - _expf(-48/9*L_a));
	
	//float3 XYZw = DWG_to_XYZ(make_float3(referenceLuminance, referenceLuminance, referenceLuminance));	// = 95.04559300, 100.00000000, 108.90577500
	float Izw = DWG_to_ZCAM_Izazbz(make_float3(referenceLuminance, referenceLuminance, referenceLuminance)).x;
	float Qzw = 2700 * spow(Izw, 1.6*F_s/_powf(F_b, 0.12)) * _powf(F_s, 2.2) * _powf(F_b, 0.5) * _powf(F_L, 0.2);
	
	// Step 1
	float Iz = spow(JzMzhz.x * Qzw / (2700 * 100 * _powf(F_s, 2.2) * _powf(F_b, 0.5) * _powf(F_L, 0.2)), _powf(F_b, 0.12)/(1.6*F_s));
	
	// Step 4
	float ez = 1.015 + _cosf((89.038 + JzMzhz.z) * PI/180);
	float Cz = spow(JzMzhz.y * (_powf(Izw, 0.78) * _powf(F_b, 0.1)) / (100 * _powf(ez, 0.068) * _powf(F_L, 0.2)), 0.5/0.37);
	float az = Cz * _cosf(JzMzhz.z * PI/180);
	float bz = Cz * _sinf(JzMzhz.z * PI/180);
	
	// Step 5
	float3 Izazbz = make_float3(Iz, az, bz);
	
	return ZCAM_Izazbz_to_DWG(Izazbz);
}

__DEVICE__ float3 DWG_to_JzCzhz(float3 rgb)
{
	float b = 1.15;
	float g = 0.66;
	float c1 = 3424 / _exp2f(12);
	float c2 = 2413 / _exp2f(7);
	float c3 = 2392 / _exp2f(7);
	float eta = 2610 / _exp2f(14);
	float rho = 1.7 * 2523 / _exp2f(5);
	float d = -0.56;
	float d0 = 1.6295499532821566 * _powf(10, -11);
	
	float3 xyz = DWG_to_XYZ(rgb);
	float3 lms, lms_;
	
	float X_ = b * xyz.x - (b - 1) * xyz.z;
	float Y_ = g * xyz.y - (g - 1) * xyz.x;

	lms.x = 0.41478972 * X_ + 0.579999 * Y_ + 0.0146480 * xyz.z;
	lms.y = -0.2015100 * X_ + 1.120649 * Y_ + 0.0531008 * xyz.z;
	lms.z = -0.0166008 * X_ + 0.264800 * Y_ + 0.6684799 * xyz.z;
	
	lms_.x = spow((c1 + c2 * spow(lms.x/10000, eta)) / (1 + c3 * spow(lms.x/10000, eta)), rho);
	lms_.y = spow((c1 + c2 * spow(lms.y/10000, eta)) / (1 + c3 * spow(lms.y/10000, eta)), rho);
	lms_.z = spow((c1 + c2 * spow(lms.z/10000, eta)) / (1 + c3 * spow(lms.z/10000, eta)), rho);
	
	float Iz = 0.5 * lms_.x + 0.5 * lms_.y;
	float az = 3.524000 * lms_.x + -4.066708 * lms_.y + 0.542708 * lms_.z;
	float bz = 0.199076 * lms_.x + 1.096799 * lms_.y + -1.295875 * lms_.z;
	
	float Jz = (1 + d) * Iz / (1 + d * Iz) - d0;
	float Cz = _sqrtf(az*az + bz*bz);
	float hz = _atan2f(bz, az);
	
	return make_float3(Jz, Cz, hz);
}

__DEVICE__ float3 JzCzhz_to_DWG(float3 JzCzhz)
{
	float b = 1.15;
	float g = 0.66;
	float c1 = 3424 / _exp2f(12);
	float c2 = 2413 / _exp2f(7);
	float c3 = 2392 / _exp2f(7);
	float eta = 2610 / _exp2f(14);
	float rho = 1.7 * 2523 / _exp2f(5);
	float d = -0.56;
	float d0 = 1.6295499532821566 * _powf(10, -11);
	
	float3 lms, lms_, xyz_, xyz;
	
	float Iz = (JzCzhz.x + d0) / (1 + d - d * (JzCzhz.x + d0));
	float az = JzCzhz.y * _cosf(JzCzhz.z);
	float bz = JzCzhz.y * _sinf(JzCzhz.z);
	
	lms_.x = 1.00000000 * Iz + 0.13860504 * az + 0.05804732 * bz;
	lms_.y = 1.00000000 * Iz + -0.13860504 * az + -0.05804732 * bz;
	lms_.z = 1.00000000 * Iz + -0.09601924 * az + -0.81189190 * bz;
	
	lms.x = 10000 * spow((c1 - spow(lms_.x, 1/rho)) / (c3 * spow(lms_.x, 1/rho) - c2), 1/eta);
	lms.y = 10000 * spow((c1 - spow(lms_.y, 1/rho)) / (c3 * spow(lms_.y, 1/rho) - c2), 1/eta);
	lms.z = 10000 * spow((c1 - spow(lms_.z, 1/rho)) / (c3 * spow(lms_.z, 1/rho) - c2), 1/eta);
	
	xyz_.x = 1.92422644 *lms.x + -1.00479231 *lms.y + 0.03765140 *lms.z;
	xyz_.y = 0.35031676 *lms.x + 0.72648119 *lms.y + -0.06538442 *lms.z;
	xyz_.z = -0.09098281 *lms.x + -0.31272829 *lms.y + 1.52276656 *lms.z;
	
	xyz.x = (xyz_.x + (b - 1) * xyz_.z) / b;
	xyz.y = (xyz_.y + (g - 1) * xyz.x) / g;
	xyz.z = xyz_.z;
	
	return XYZ_to_DWG(xyz);
}

__DEVICE__ float3 DWG2ICh(float3 rgb)
{
	float3 xyz, lms, lms_, ipt;
	
	// DaVinci Wide Gamut to XYZ D65
	xyz = DWG_to_XYZ(rgb);
	
	// XYZ to LMS
	lms.x = 0.4002*xyz.x + 0.7075*xyz.y + -0.0807*xyz.z;
	lms.y = -0.2280*xyz.x + 1.150*xyz.y +  0.0612*xyz.z;
	lms.z = 0.9184*xyz.z;
	
	// LMS to LMS'
	lms_.x = _copysignf(1, lms.x) * _powf(_fabs(lms.x), 0.43);
	lms_.y = _copysignf(1, lms.y) * _powf(_fabs(lms.y), 0.43);
	lms_.z = _copysignf(1, lms.z) * _powf(_fabs(lms.z), 0.43);
	
	// LMS' to IPT
	ipt.x = 0.4000*lms_.x + 0.4000*lms_.y + 0.2000*lms_.z;
	ipt.y = 4.4550*lms_.x +-4.8510*lms_.y + 0.3960*lms_.z;
	ipt.z = 0.8056*lms_.x + 0.3572*lms_.y +-1.1628*lms_.z;
	
	// IPT to LCh
	float C = _sqrtf(ipt.y*ipt.y + ipt.z*ipt.z);
	float h = _atan2f(ipt.z, ipt.y);
	
	return make_float3(ipt.x, C, h);
}

__DEVICE__ float3 ICh2DWG(float3 ICh)
{
	float3 ipt, lms_, lms, xyz;
	
	// LCh to IPT
	ipt.x = ICh.x;
	ipt.y = ICh.y * _cosf(ICh.z);
	ipt.z = ICh.y * _sinf(ICh.z);
	
	// IPT to LMS'
	lms_.x = 1.00000000*ipt.x + 0.09756893*ipt.y + 0.20522643*ipt.z;
	lms_.y = 1.00000000*ipt.x + -0.11387649*ipt.y + 0.13321716*ipt.z;
	lms_.z = 1.00000000*ipt.x + 0.03261511*ipt.y + -0.67688718*ipt.z;
	
	// LMS' to LMS
	lms.x = _copysignf(1, lms_.x) * _powf(_fabs(lms_.x), 1/0.43);
	lms.y = _copysignf(1, lms_.y) * _powf(_fabs(lms_.y), 1/0.43);
	lms.z = _copysignf(1, lms_.z) * _powf(_fabs(lms_.z), 1/0.43);
	
	// LMS to XYZ
	xyz.x = 1.850243*lms.x + -1.138302*lms.y + 0.238435*lms.z;
	xyz.y = 0.366831*lms.x + 0.643885*lms.y + -0.010673*lms.z;
	xyz.z = 1.088850*lms.z;
	
	// XYZ to DaVinci Wide Gamut
	return XYZ_to_DWG(xyz);
}

__DEVICE__ float3 DWG2LCh(float3 rgb)
{
	// D65 reference white point
	float Xn = 0.950456;	float Yn = 1.0;	float Zn = 1.088754;
	
	// sRGB D65 to XYZ D65
	// float X = 0.4124564*rgb.x + 0.3575761*rgb.y + 0.1804375*rgb.z;
	// float Y = 0.2126729*rgb.x + 0.7151522*rgb.y + 0.0721750*rgb.z;
	// float Z = 0.0193339*rgb.x + 0.1191920*rgb.y + 0.9503041*rgb.z;
	
	// DaVinci Wide Gamut to XYZ D65
	float X = 0.70062239*rgb.x + 0.14877482*rgb.y + 0.10105872*rgb.z;
	float Y = 0.27411851*rgb.x + 0.87363190*rgb.y + -0.14775041*rgb.z;
	float Z = -0.09896291*rgb.x + -0.13789533*rgb.y + 1.32591599*rgb.z;
	
	// XYZ D65 to Lab
	float exprX = (X/Xn <= 216.0/24389) ? 1.0/116 * (24389/27 * X/Xn + 16) : powf(X/Xn, 1.0/3);
	float exprY = (Y/Yn <= 216.0/24389) ? 1.0/116 * (24389/27 * Y/Yn + 16) : powf(Y/Yn, 1.0/3);
	float exprZ = (Z/Zn <= 216.0/24389) ? 1.0/116 * (24389/27 * Z/Zn + 16) : powf(Z/Zn, 1.0/3);
	float L = 116 * exprY - 16;
	float a = 500 * (exprX - exprY);
	float b = 200 * (exprY - exprZ);
	
	// Lab to LCh
	float C = sqrt(a*a + b*b);
	float h = atan2f(b, a);
	
	return make_float3(L, C, h);
}

__DEVICE__ float3 LCh2sDWG(float3 LCh)
{
	float L = LCh.x;	float C = LCh.y;	float h = LCh.z;
	
	// D65 reference white point
	float Xr = 0.950456;	float Yr = 1.0;	float Zr = 1.088754;
	
	// LCh to Lab
	float a = C * cosf(h);
	float b = C * sinf(h);
	
	// Lab to XYZ
	float exprY = (L + 16) / 116;
	float exprX = a/500 + exprY;
	float exprZ = exprY - b/200;
	
	float x_r = ( powf(exprX, 3) > 216.0/24389 ) ? powf(exprX, 3) : (116*exprX - 16) / (24389/27);
	float y_r = ( L > 216.0/27 ) ? powf( (L + 16) / 116, 3) : L / (24389/27);
	float z_r = ( powf(exprZ, 3) > 216.0/24389 ) ? powf(exprZ, 3) : (116*exprZ - 16) / (24389/27);
	
	float X = x_r * Xr;
	float Y = y_r * Yr;
	float Z = z_r * Zr;
	
	// XYZ D65 to sRGB D65
	// float R = 3.2404542*X - 1.5371385*Y - 0.4985314*Z;
	// float G = -0.9692660*X + 1.8760108*Y + 0.0415560*Z;
	// float B = 0.0556434*X - 0.2040259*Y + 1.0572252*Z;
	
	// XYZ D65 to DaVinci Wide Gamut
	float R = 1.51667204*X -0.28147805 *Y -0.14696363*Z;
	float G = -0.46491710*X + 1.25142378*Y + 0.17488461*Z;
	float B = 0.06484905*X + 0.10913934*Y + 0.76141462*Z;
	
	return make_float3(R, G, B);
}