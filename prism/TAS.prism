dtmc
//goal of probabilistic model checking
//P1. verify the probability pTAS of successfully invoking the TAS workflow
//P2. verify the expected cost cTAS of invoking the TAS workflow


//parameters measure the reliability + cost of a concrete service
//Alarm is the interface (e.g. sendAlarm) this is called an abstract service
//different concrete service providers that offer sendAlarm
//alarm1 = (pAlarm = 0.98,cAlarm = 1.25)
//alarm2 = (pAlarm = 0.91,cAlarm = 1.10)
//alarm3 = ()
//analaysis1, analysis2, analysis3, analysis4

//choose the best configuration to fit a particular user profile (e.g. values for pChangeResult, pVitalParamsPicked, buttonMsgPicked)
//user profiles: elderly, middle-age, young

//choose the best configuration over a dynamically changing concrete service
//imagine that you picked (alarm1,analysis3,drug2) but drug2 becomes too expensive. or alarm1 becomes unreliable.

//question is: given a collection of concrete services for each abstract service
//can we choose the best configuration which maps abstract -> concrete service
//best means: reward: high reliability + low cost.
//max pTAS and  min cTAS

//maximise expected reward E = w1 * pTAS + w2 * (-1)*cTAS
//how do we obtain an algebraic expression of the TAS workflow?
//parametric model checking.

//action selection: choose a concrete service which maximises E.
//e.g. revaluate E every time we choose an action.

//use reinforcement learning which will select concrete services (using multi-armed bandit approach)
//and find the

//parameters
//probabilities measuring reliability for each TAS service
//pAnalysis = probability that the medical analysis service is invoked successfully
//pDrug = probability that the drug service is invoked successfully
//pAlarm = probability that the alarm service is invoked successfully


//state space is assigning a value to each one of these probabilities
//which in turn impacts the performance (expected reward) outcome

const double pAnalysis;
const double cAnalysis;
const double pDrug;
const double cDrug;
const double pAlarm;
const double cAlarm;

//analysis result parameters
//change either drug or dose
//TASK 1. add patientOK -> meaning that analysis succeeds and the workflow succeeds.
//TASK 2. apply parametric model checking to this model.
// - remove probability value assignments from the parameters
// - run pmc using Prism from the command line.
// - prism updatedtas.pm properties.pctl -param  'pAnalysis=0.0:1.0,pDrug=0.0:1.0'
const double pChangeResult;
const double pAlarmResult  = 1 - pChangeResult;
const double pPatientOK = 1 - pChangeResult - pAlarmResult;

const double pVitalParamsPicked;
const double buttonMsgPicked    = 1-pVitalParamsPicked;

//state constants for readability.
const int READY=0;
const int ANALYSIS=1;
const int ALARM = 2;
const int ALARM_SUCC = 3;
const int ANALYSIS_RESULTS = 4;
const int DRUG = 5;
const int DRUG_SUCC = 6;
const int SUCC = 7;
const int DONE = 8;

const int FAIL = 10;

module TAS
state : [0..10] init READY;

[pick] (state=READY) -> pVitalParamsPicked:(state'=ANALYSIS) + (buttonMsgPicked):(state'=ALARM);


//invoke the analysis service,
[medicalAnalysis] (state =ANALYSIS) -> pAnalysis:(state'=ANALYSIS_RESULTS) + (1-pAnalysis):(state'=FAIL);

[analysisResult] (state=ANALYSIS_RESULTS) -> pPatientOK:(state' = SUCC) + pChangeResult:(state'=DRUG) + (pAlarmResult):(state'=ALARM);

//invoke the drug service
[drug] (state=DRUG) ->  pDrug:(state'=DRUG_SUCC) + (1-pDrug):(state'=FAIL);

//invoke the alarm service, which succeeds or fails.
[alarm] (state=ALARM) -> pAlarm:(state'=ALARM_SUCC) + (1-pAlarm):(state'=FAIL);

[alarm_success] (state = ALARM_SUCC) -> 1.0:(state'=SUCC);
[drug_success] (state = DRUG_SUCC) -> 1.0:(state'=SUCC);

[success] (state = SUCC) -> (state'=DONE);
[fail] (state = FAIL) -> (state'=DONE);

[done] (state = DONE) -> true;

endmodule

rewards
(state = ANALYSIS) : cAnalysis;
(state = ALARM) : cAlarm;
(state = DRUG) : cDrug;
endrewards
