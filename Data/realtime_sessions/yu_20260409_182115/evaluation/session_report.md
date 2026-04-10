# Session Report

- samples_with_reference: 144
- total_samples: 147
- evaluation_filters: abs_time_delta_ms<=350.0, output_valid=true, reject_reason=ok, artifact_flag=0, ref_pp_inlier
- plot_series: smoothed

## Smoothed Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",56,6.417885884129065,6.757150383570107,-0.2910168527642183,-6.417885884129065,4.249138850544911,4.0303396588769544
RTBP,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",56,10.44822554300602,10.576180314721416,0.0573082968671033,-10.44822554300602,4.249138850544911,4.0303396588769544
SinBP_D,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",52,2.0808382684971884,2.5374239030373897,-0.1420807149750689,-1.599434613249573,5.0944673948606,4.911335569868742
SinBP_D,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",52,6.510770183118314,6.721041720311483,-0.0937116249212188,-6.510770183118314,5.0944673948606,4.911335569868742
SinBP_D_EOnly,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,3.686202238653158,4.085670578945996,-0.1257320060403463,-3.5056415175326183,5.0513619837352195,4.902744161565473
SinBP_D_EOnly,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,8.40838567909809,8.566689036571766,-0.2303880828240868,-8.40838567909809,5.0513619837352195,4.902744161565473
SinBP_D_E2,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,4.606069476859751,5.017518610220025,-0.155486556680533,-4.535103637270281,4.691050878023492,4.485580172200022
SinBP_D_E2,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,9.020683809470302,9.16391865300688,-0.216209647855173,-9.020683809470302,4.691050878023492,4.485580172200022
SinBP_D_LocalA,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,2.2042518305996235,2.584738333318872,-0.0865389146242311,-1.5781033573317276,4.919330149381865,4.750125129994771
SinBP_D_LocalA,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,6.3282284873264985,6.553166832986431,-0.2500346028053495,-6.3282284873264985,4.919330149381865,4.750125129994771
SinBP_M,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,3.670760092990604,4.130735232817296,-0.1484672383929133,-3.48138899544679,2.292755338877447,1.0649985501745989
SinBP_M,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,4.546387545621389,4.77712587963729,0.1319538874352073,-4.546387545621389,2.292755338877447,1.0649985501745989


## Calibrated Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",56,6.417885884129065,6.757150383570107,-0.2910168527642183,-6.417885884129065,4.249138850544911,4.0303396588769544
RTBP,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",56,10.44822554300602,10.576180314721416,0.0573082968671033,-10.44822554300602,4.249138850544911,4.0303396588769544
SinBP_D,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",52,2.0808382684971884,2.5374239030373897,-0.1420807149750689,-1.599434613249573,5.0944673948606,4.911335569868742
SinBP_D,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",52,6.510770183118314,6.721041720311483,-0.0937116249212188,-6.510770183118314,5.0944673948606,4.911335569868742
SinBP_D_EOnly,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,3.686202238653158,4.085670578945996,-0.1257320060403463,-3.5056415175326183,5.0513619837352195,4.902744161565473
SinBP_D_EOnly,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,8.40838567909809,8.566689036571766,-0.2303880828240868,-8.40838567909809,5.0513619837352195,4.902744161565473
SinBP_D_E2,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,4.606069476859751,5.017518610220025,-0.155486556680533,-4.535103637270281,4.691050878023492,4.485580172200022
SinBP_D_E2,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,9.020683809470302,9.16391865300688,-0.216209647855173,-9.020683809470302,4.691050878023492,4.485580172200022
SinBP_D_LocalA,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,2.2042518305996235,2.584738333318872,-0.0865389146242311,-1.5781033573317276,4.919330149381865,4.750125129994771
SinBP_D_LocalA,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,6.3282284873264985,6.553166832986431,-0.2500346028053495,-6.3282284873264985,4.919330149381865,4.750125129994771
SinBP_M,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,3.670760092990604,4.130735232817296,-0.1484672383929133,-3.48138899544679,2.292755338877447,1.0649985501745989
SinBP_M,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,4.546387545621389,4.77712587963729,0.1319538874352073,-4.546387545621389,2.292755338877447,1.0649985501745989


## Raw Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",56,6.716002521875001,7.450021729355021,-0.2231969812844328,-6.7044831701607155,4.389032432589283,4.134439847803569
RTBP,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",56,10.838923017964284,11.363615316962358,0.096559678594961,-10.838923017964284,4.389032432589283,4.134439847803569
SinBP_D,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",52,2.5230499786153846,3.053574757672172,-0.2077523276769925,-1.6967098345769231,5.306555756076923,5.034352582923077
SinBP_D,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",52,6.7353977670769245,7.285015145950408,0.0501369918553673,-6.731062417500002,5.306555756076923,5.034352582923077
SinBP_D_EOnly,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,3.900935391854167,4.454812937440286,0.0686668476503288,-3.4440254155208336,5.183948354520832,4.971253665729166
SinBP_D_EOnly,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,8.41527908125,8.976965528611842,-0.2264421612628817,-8.41527908125,5.183948354520832,4.971253665729166
SinBP_D_E2,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,4.768278775770833,5.351547817619794,0.0651839267164414,-4.4802754155208335,4.844213709645833,4.5522953323958335
SinBP_D_E2,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,9.032570747916669,9.568989162376486,-0.2235021999780534,-9.032570747916669,4.844213709645833,4.5522953323958335
SinBP_D_LocalA,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,2.657777932854167,3.10179542297366,0.0742847303508036,-1.522983748854167,5.057050138479166,4.819378665729166
SinBP_D_LocalA,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",48,6.342362414583334,7.040196202908003,-0.2319212775553256,-6.342362414583334,5.057050138479166,4.819378665729166
SinBP_M,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,4.172462640571429,4.8616217528571575,-0.2491018894441788,-3.830354798380952,2.434546385476191,1.1690527393809524
SinBP_M,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,5.070700187761905,5.6073472547223915,0.2047719069265641,-4.999407537761905,2.434546385476191,1.1690527393809524


## Plots

- sbp_timeseries.png
- dbp_timeseries.png

