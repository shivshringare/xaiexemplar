Memory limits: cudd=1g, java(heap)=1g
Command line: prism TAS.prism tasproperties.pctl -param 'pAnalysis=0.0:1.0,pDrug=0.0:1.0,cAnalysis=0.0:1.0,cDrug=0.0:1.0,pAlarm=0.0:1.0,cAlarm=0.0:1.5,pChangeResult=0.0:1.0,pVitalParamsPicked=0.0:1.0'

Parsing model file "TAS.prism"...

Type:        DTMC
Modules:     TAS
Variables:   state

Parsing properties file "tasproperties.pctl"...

2 properties:
(1) P=? [ F (state=SUCC) ]
(2) R=? [ F (state=DONE) ]

---------------------------------------------------------------------

Parametric model checking: P=? [ F (state=SUCC) ]

Building model (parametric engine)...

Computing reachable states...
Reachable states exploration and model construction done in 0.031 secs.

States:      10 (1 initial)
Transitions: 15

Time for model construction: 0.031 seconds.

Time for model checking: 0.078 seconds.

Result (probability): ([0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.5],[0.0,1.0],[0.0,1.0]): { pVitalParamsPicked * pChangeResult * pDrug * pAnalysis - pVitalParamsPicked * pChangeResult * pAlarm * pAnalysis + pVitalParamsPicked * pAlarm * pAnalysis - pVitalParamsPicked * pAlarm + pAlarm }

---------------------------------------------------------------------

Parametric model checking: R=? [ F (state=DONE) ]

Building model (parametric engine)...

Computing reachable states...
Reachable states exploration and model construction done in 0.015 secs.

States:      10 (1 initial)
Transitions: 15

Time for model construction: 0.015 seconds.
Building reward structure...

Time for model checking: 0.063 seconds.

Result (expected reward): ([0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.5],[0.0,1.0],[0.0,1.0]): { pVitalParamsPicked * pChangeResult * cDrug * pAnalysis - pVitalParamsPicked * pChangeResult * cAlarm * pAnalysis + pVitalParamsPicked * cAlarm * pAnalysis + pVitalParamsPicked * cAnalysis - pVitalParamsPicked * cAlarm + cAlarm }
