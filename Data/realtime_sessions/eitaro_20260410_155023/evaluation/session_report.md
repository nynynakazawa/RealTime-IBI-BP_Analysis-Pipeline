# Session Report

- samples_with_reference: 300
- total_samples: 300
- evaluation_filters: abs_time_delta_ms<=350.0, output_valid=true, reject_reason=ok, artifact_flag=0, ref_pp_inlier
- plot_series: smoothed

## Smoothed Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",213,12.36849288296651,12.674485510272095,-0.2547915403816084,12.36849288296651,18.48903353301412,18.48903353301412
RTBP,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",213,6.1205406500476105,6.279070625591832,-0.263298581639375,-6.1205406500476105,18.48903353301412,18.48903353301412
SinBP_D,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",208,12.690701187745164,13.004151111504992,-0.3664772448838779,12.690701187745164,18.968002767288212,18.968002767288212
SinBP_D,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",208,6.2773015795430505,6.4389300031745105,-0.3721949871569104,-6.2773015795430505,18.968002767288212,18.968002767288212
SinBP_D_EOnly,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,28.337467119514777,28.458263629951983,-0.2668736555409128,28.337467119514777,20.159236847064943,20.159236847064943
SinBP_D_EOnly,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,8.17823027244983,8.369170183818706,-0.3339538859558715,8.17823027244983,20.159236847064943,20.159236847064943
SinBP_D_E2,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,27.352224028481977,27.45252336342496,-0.1874436471268213,27.352224028481977,19.758324557277952,19.758324557277952
SinBP_D_E2,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,7.5938994712040255,7.805644929476204,-0.4201186252936846,7.5938994712040255,19.758324557277952,19.758324557277952
SinBP_D_LocalA,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,30.516216924067635,30.631209162126822,-0.307889482791253,30.516216924067635,19.942433280997328,19.942433280997328
SinBP_D_LocalA,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,10.573783643070309,10.688138509261709,-0.3809975873996014,10.573783643070309,19.942433280997328,19.942433280997328
SinBP_M,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",188,12.373507594979335,12.640325496195125,-0.2312404258342314,12.373507594979335,18.477305382269307,18.477305382269307
SinBP_M,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",188,6.1037977872899685,6.239818982452102,-0.2354045614507488,-6.1037977872899685,18.477305382269307,18.477305382269307


## Calibrated Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",213,12.36849288296651,12.674485510272095,-0.2547915403816084,12.36849288296651,18.48903353301412,18.48903353301412
RTBP,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",213,6.1205406500476105,6.279070625591832,-0.263298581639375,-6.1205406500476105,18.48903353301412,18.48903353301412
SinBP_D,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",208,12.690701187745164,13.004151111504992,-0.3664772448838779,12.690701187745164,18.968002767288212,18.968002767288212
SinBP_D,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",208,6.2773015795430505,6.4389300031745105,-0.3721949871569104,-6.2773015795430505,18.968002767288212,18.968002767288212
SinBP_D_EOnly,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,28.337467119514777,28.458263629951983,-0.2668736555409128,28.337467119514777,20.159236847064943,20.159236847064943
SinBP_D_EOnly,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,8.17823027244983,8.369170183818706,-0.3339538859558715,8.17823027244983,20.159236847064943,20.159236847064943
SinBP_D_E2,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,27.352224028481977,27.45252336342496,-0.1874436471268213,27.352224028481977,19.758324557277952,19.758324557277952
SinBP_D_E2,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,7.5938994712040255,7.805644929476204,-0.4201186252936846,7.5938994712040255,19.758324557277952,19.758324557277952
SinBP_D_LocalA,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,30.516216924067635,30.631209162126822,-0.307889482791253,30.516216924067635,19.942433280997328,19.942433280997328
SinBP_D_LocalA,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,10.573783643070309,10.688138509261709,-0.3809975873996014,10.573783643070309,19.942433280997328,19.942433280997328
SinBP_M,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",188,12.373507594979335,12.640325496195125,-0.2312404258342314,12.373507594979335,18.477305382269307,18.477305382269307
SinBP_M,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",188,6.1037977872899685,6.239818982452102,-0.2354045614507488,-6.1037977872899685,18.477305382269307,18.477305382269307


## Raw Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",213,12.38676641310798,12.720400599334234,-0.2614705245210895,12.38676641310798,18.51626735319249,18.51626735319249
RTBP,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",213,6.129500940084506,6.302366874598859,-0.2687785680612107,-6.129500940084506,18.51626735319249,18.51626735319249
SinBP_D,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",208,12.666466443471151,13.008791297608663,-0.3514328673544437,12.666466443471151,18.93130851002404,18.93130851002404
SinBP_D,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",208,6.264842066552884,6.441575911611737,-0.3583866280036383,-6.264842066552884,18.93130851002404,18.93130851002404
SinBP_D_EOnly,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,28.393199976931708,28.58054695697068,-0.2108714584904419,28.393199976931708,20.162517511848783,20.14556212000488
SinBP_D_EOnly,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,8.24763785692683,8.491649711464037,-0.3010609597648017,8.24763785692683,20.162517511848783,20.14556212000488
SinBP_D_E2,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,27.400809733029263,27.54612771998675,-0.1500511711991261,27.400809733029263,19.744537729760975,19.744537729760975
SinBP_D_E2,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,7.656272003268294,7.9279416581384625,-0.3866404672320769,7.656272003268294,19.744537729760975,19.744537729760975
SinBP_D_LocalA,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,30.569590220834147,30.732283050967514,-0.2619678521368902,30.569590220834147,19.958858975263414,19.93439138829756
SinBP_D_LocalA,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",205,10.635198832536586,10.76896423122206,-0.3560531496295793,10.635198832536586,19.958858975263414,19.93439138829756
SinBP_M,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",188,12.346393921941493,12.67366423172429,-0.2011430480824619,12.346393921941493,18.437005781888296,18.437005781888296
SinBP_M,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",188,6.090611859946806,6.258316857505375,-0.209060339480429,-6.090611859946806,18.437005781888296,18.437005781888296


## Plots

- sbp_timeseries.png
- dbp_timeseries.png

