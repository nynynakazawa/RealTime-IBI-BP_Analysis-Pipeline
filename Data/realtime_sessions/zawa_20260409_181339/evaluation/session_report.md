# Session Report

- samples_with_reference: 190
- total_samples: 203
- evaluation_filters: abs_time_delta_ms<=350.0, output_valid=true, reject_reason=ok, artifact_flag=0, ref_pp_inlier
- plot_series: smoothed

## Smoothed Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",41,9.153484055045434,9.503002613246831,-0.0135727495583363,9.112569040848948,19.024335961316183,19.024335961316183
RTBP,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",41,9.911766920467237,10.098612411698296,0.0003039478702809,-9.911766920467237,19.024335961316183,19.024335961316183
SinBP_D,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",40,13.273691922296114,13.544591750175782,-0.092576731498193,13.273691922296114,20.603907037100846,20.603907037100846
SinBP_D,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",40,7.330215114804728,7.60809557888811,-0.0491416628597003,-7.330215114804728,20.603907037100846,20.603907037100846
SinBP_D_EOnly,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,11.648198018887856,11.978799118247313,-0.0068051703171987,11.648198018887856,20.62468877116141,20.62468877116141
SinBP_D_EOnly,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,8.976490752273556,9.189255377456751,-0.0016962402255724,-8.976490752273556,20.62468877116141,20.62468877116141
SinBP_D_E2,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,10.903129497494051,11.267595489469132,-0.0027659738113201,10.903129497494051,20.32059072772453,20.32059072772453
SinBP_D_E2,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,9.41746123023048,9.617268847945684,-0.0029006255871645,-9.41746123023048,20.32059072772453,20.32059072772453
SinBP_D_LocalA,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,13.340587846630246,13.626400904419956,-0.0108954847757032,13.340587846630246,20.51315621016969,20.51315621016969
SinBP_D_LocalA,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,7.172568363539441,7.4461958877443,0.0024077644385437,-7.172568363539441,20.51315621016969,20.51315621016969
SinBP_M,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,10.894493526280405,11.273572056251734,-0.2309012864588731,10.894493526280405,17.634716547384414,17.634716547384414
SinBP_M,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,6.740223021104008,6.846594134785903,0.3515862139080027,-6.740223021104008,17.634716547384414,17.634716547384414


## Calibrated Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",41,9.153484055045434,9.503002613246831,-0.0135727495583363,9.112569040848948,19.024335961316183,19.024335961316183
RTBP,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",41,9.911766920467237,10.098612411698296,0.0003039478702809,-9.911766920467237,19.024335961316183,19.024335961316183
SinBP_D,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",40,13.273691922296114,13.544591750175782,-0.092576731498193,13.273691922296114,20.603907037100846,20.603907037100846
SinBP_D,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",40,7.330215114804728,7.60809557888811,-0.0491416628597003,-7.330215114804728,20.603907037100846,20.603907037100846
SinBP_D_EOnly,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,11.648198018887856,11.978799118247313,-0.0068051703171987,11.648198018887856,20.62468877116141,20.62468877116141
SinBP_D_EOnly,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,8.976490752273556,9.189255377456751,-0.0016962402255724,-8.976490752273556,20.62468877116141,20.62468877116141
SinBP_D_E2,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,10.903129497494051,11.267595489469132,-0.0027659738113201,10.903129497494051,20.32059072772453,20.32059072772453
SinBP_D_E2,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,9.41746123023048,9.617268847945684,-0.0029006255871645,-9.41746123023048,20.32059072772453,20.32059072772453
SinBP_D_LocalA,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,13.340587846630246,13.626400904419956,-0.0108954847757032,13.340587846630246,20.51315621016969,20.51315621016969
SinBP_D_LocalA,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,7.172568363539441,7.4461958877443,0.0024077644385437,-7.172568363539441,20.51315621016969,20.51315621016969
SinBP_M,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,10.894493526280405,11.273572056251734,-0.2309012864588731,10.894493526280405,17.634716547384414,17.634716547384414
SinBP_M,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,6.740223021104008,6.846594134785903,0.3515862139080027,-6.740223021104008,17.634716547384414,17.634716547384414


## Raw Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",41,9.502883322365854,10.2796322026671,-0.0472632353360047,9.470015568609757,18.992351047926828,18.992351047926828
RTBP,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",41,9.711293078243903,10.225502669618896,0.0479574136738465,-9.522335479317071,18.992351047926828,18.992351047926828
SinBP_D,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",40,13.54698278165,14.101197338595076,-0.0402434812249544,13.54698278165,20.542228739075,20.542228739075
SinBP_D,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",40,7.510505422324999,8.155236283443514,-0.0655016014050569,-6.995245957425001,20.542228739075,20.542228739075
SinBP_D_EOnly,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,11.952152553184211,12.726720212772149,-0.0427211392021264,11.952152553184211,20.58022611521053,20.58022611521053
SinBP_D_EOnly,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,8.96879097139474,9.523862818889352,0.0457927384287741,-8.62807356202632,20.58022611521053,20.58022611521053
SinBP_D_E2,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,11.184520974236843,12.038686086696744,-0.0411794574756205,11.184520974236843,20.269436641526315,20.269436641526315
SinBP_D_E2,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,9.403001497710529,9.936378192982188,0.0463782535887568,-9.084915667289472,20.269436641526315,20.269436641526315
SinBP_D_LocalA,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,13.655047290026316,14.323421989818138,-0.0436887199730039,13.655047290026316,20.466805062578953,20.466805062578953
SinBP_D_LocalA,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",38,7.32672579144737,7.927198566941265,0.0455045114415204,-6.8117577725526335,20.466805062578953,20.466805062578953
SinBP_M,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,10.762204596095238,11.308605098761442,-0.1508799918713276,10.762204596095238,17.605203183999997,17.605203183999997
SinBP_M,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",42,6.842998587904762,7.215719185045838,0.1368225355168312,-6.842998587904762,17.605203183999997,17.605203183999997


## Plots

- sbp_timeseries.png
- dbp_timeseries.png

