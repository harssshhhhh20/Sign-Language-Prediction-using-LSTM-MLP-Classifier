| Feature group | Dims | % of full | Accuracy | Macro-F1 |
|---|---|---|---|---|
| full | 1662 | 100.0 | 95.9 ± 6.5 | 95.8 ± 6.6 |
| face | 1404 | 84.5 | 79.5 ± 12.2 | 77.7 ± 13.9 |
| pose_hands | 258 | 15.5 | 98.2 ± 3.3 | 98.2 ± 3.3 |
| upper_pose_hands | 226 | 13.6 | 98.4 ± 3.0 | 98.4 ± 3.0 |
| articulator_hands | 150 | 9.0 | 98.4 ± 3.0 | 98.4 ± 3.0 |
| hands | 126 | 7.6 | 98.5 ± 2.4 | 98.5 ± 2.4 |
| dominant_hand | 63 | 3.8 | 99.6 ± 1.2 | 99.6 ± 1.2 |

_Feature-group ablation. Dimensionality is per frame. The `face` row is a control: manual signs cannot be distinguished by face landmarks, so accuracy above chance there measures leakage rather than recognition._
