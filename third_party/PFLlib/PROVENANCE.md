# Imported PFLlib subset

Source: https://github.com/TsingZ0/PFLlib
Upstream revision: 0169ba7e412c9856a08bb3faefab1e35f538a3c1.
Exported with git archive from an existing local upstream remote ref; Git normalizes line endings on commit.
Apache-2.0 license retained in LICENSE. No local donor modifications imported.
Only FedAvg's required server/client/model/data/DLG modules and dataset utilities
are included. TTFL entrypoints import these modules; upstream files are unchanged.

The upstream FedAvg loop uses range(global_rounds+1). TTFL sets this argument to
requested_rounds-1 so 100 requested rounds produce exactly100 aggregation updates.
The upstream method evaluates before updating; our final evaluation explicitly
broadcasts the saved final global model before scoring.

