# Session Report

- samples_with_reference: 149
- total_samples: 152
- evaluation_filters: abs_time_delta_ms<=350.0, output_valid=true, reject_reason=ok, artifact_flag=0, ref_pp_inlier
- plot_series: smoothed

## Smoothed Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",101,4.8702089514160285,6.072109382364773,-0.2073248502061295,-3.873040115008191,4.182231538646883,-0.4521839613048405
RTBP,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",101,3.500892514505749,4.356090282869489,0.418915144118865,-3.4208561537033506,4.182231538646883,-0.4521839613048405
SinBP_D,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",93,3.831751748702524,4.71618176842068,-0.2265410270698363,0.7758486806211726,4.325904257126447,-0.4051905688528315
SinBP_D,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",93,2.355706912871976,2.9528279925953864,0.4889791578307333,1.1810392494740043,4.325904257126447,-0.4051905688528315
SinBP_D_EOnly,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,3.4891920676911554,4.512710182839121,-0.1356228774314515,-0.9901819190216616,4.464601598138331,1.0656807456594122
SinBP_D_EOnly,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,2.8141614802981443,3.835663445479848,0.3238236451524615,-2.055862664681074,4.464601598138331,1.0656807456594122
SinBP_D_E2,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,3.5240608492523364,4.596792742719026,-0.1072733505818821,-1.57577716822865,4.4578703926514915,0.8262952321824357
SinBP_D_E2,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,2.989424720450745,3.9422699406824946,0.3293367541936075,-2.4020724004110856,4.4578703926514915,0.8262952321824357
SinBP_D_LocalA,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",88,3.8910328089638337,4.801166767050099,-0.2391763944004374,1.0081998377002372,4.4589175568117305,0.3555496834663885
SinBP_D_LocalA,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",88,2.3769364780424405,3.1099326062475376,0.3831508324083594,0.6526501542338488,4.4589175568117305,0.3555496834663885
SinBP_M,smoothed,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",80,3.2611495038739724,4.358437079211581,-0.0487843579107765,-1.145717691765839,4.319500633361369,-2.2544896274761252
SinBP_M,smoothed,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",80,2.4377740885071653,2.853789844961293,0.348132764471212,1.1087719357102863,4.319500633361369,-2.2544896274761252


## Calibrated Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",101,4.8702089514160285,6.072109382364773,-0.2073248502061295,-3.873040115008191,4.182231538646883,-0.4521839613048405
RTBP,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",101,3.500892514505749,4.356090282869489,0.418915144118865,-3.4208561537033506,4.182231538646883,-0.4521839613048405
SinBP_D,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",93,3.831751748702524,4.71618176842068,-0.2265410270698363,0.7758486806211726,4.325904257126447,-0.4051905688528315
SinBP_D,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",93,2.355706912871976,2.9528279925953864,0.4889791578307333,1.1810392494740043,4.325904257126447,-0.4051905688528315
SinBP_D_EOnly,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,3.4891920676911554,4.512710182839121,-0.1356228774314515,-0.9901819190216616,4.464601598138331,1.0656807456594122
SinBP_D_EOnly,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,2.8141614802981443,3.835663445479848,0.3238236451524615,-2.055862664681074,4.464601598138331,1.0656807456594122
SinBP_D_E2,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,3.5240608492523364,4.596792742719026,-0.1072733505818821,-1.57577716822865,4.4578703926514915,0.8262952321824357
SinBP_D_E2,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,2.989424720450745,3.9422699406824946,0.3293367541936075,-2.4020724004110856,4.4578703926514915,0.8262952321824357
SinBP_D_LocalA,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",88,3.8910328089638337,4.801166767050099,-0.2391763944004374,1.0081998377002372,4.4589175568117305,0.3555496834663885
SinBP_D_LocalA,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",88,2.3769364780424405,3.1099326062475376,0.3831508324083594,0.6526501542338488,4.4589175568117305,0.3555496834663885
SinBP_M,calibrated,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",80,3.2611495038739724,4.358437079211581,-0.0487843579107765,-1.145717691765839,4.319500633361369,-2.2544896274761252
SinBP_M,calibrated,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",80,2.4377740885071653,2.853789844961293,0.348132764471212,1.1087719357102863,4.319500633361369,-2.2544896274761252


## Raw Metrics

method,series,target,filters,n,mae,rmse,corr,signed_bias,pp_mae,pp_signed_bias
RTBP,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",101,6.134464055930694,8.125372307762367,-0.028653694127985,-4.4206145705049495,4.554742119346535,-0.3091739496435642
RTBP,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",101,5.957544340821781,8.33681091788157,0.1130034870411799,-4.111440620861385,4.554742119346535,-0.3091739496435642
SinBP_D,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",93,4.771772089322579,6.1804836407603,-0.1398095375358236,0.434499180870968,4.976781656365592,-0.2979353367096771
SinBP_D,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",93,4.58114502383871,6.036751256849418,0.2331482116517316,0.7324345175806452,4.976781656365592,-0.2979353367096771
SinBP_D_EOnly,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,5.086197101434782,6.740395635761197,0.0139830434741183,-1.1954007956956512,5.2879964086847835,1.0881334735978274
SinBP_D_EOnly,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,5.432359319945652,7.499038216617437,0.0455212589287648,-2.2835342692934786,5.2879964086847835,1.0881334735978274
SinBP_D_E2,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,5.040523453369565,6.366842395618775,0.0668621699844847,-1.7935529696086945,5.247996727445654,0.8469378214239135
SinBP_D_E2,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",92,5.521817199358696,7.519714436936978,0.0233226080748818,-2.640490791032608,5.247996727445654,0.8469378214239135
SinBP_D_LocalA,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",88,5.111735576397726,6.89887778632523,-0.0242643716940682,0.8020489552840909,4.978406980352271,0.3786917933068183
SinBP_D_LocalA,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",88,5.249927601068181,7.322175959215617,0.0566180863106799,0.4233571619772726,4.978406980352271,0.3786917933068183
SinBP_M,raw,SBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",80,5.540901242012501,7.128751634642073,-0.0450617288557305,-1.1662817292874998,4.536596163225001,-2.1787239735500004
SinBP_M,raw,DBP,"abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",80,4.4011956884875,6.081054799391069,0.1657348426532476,1.0124422442625003,4.536596163225001,-2.1787239735500004


## Plots

- sbp_timeseries.png
- dbp_timeseries.png

