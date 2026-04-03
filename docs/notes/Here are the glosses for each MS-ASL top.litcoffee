Here are the glosses for each MS-ASL top‑k run (in dictionary order, same as the LSTM output indices):

Top‑20: cousin, deaf, eat, family, father, finish, friend, here, like, milk, mother, nice, orange, pencil, school, sister, student, teacher, what, where
Top‑10: cousin, eat, family, father, finish, nice, student, teacher, what, where
Top‑6: cousin, eat, finish, nice, student, teacher
Top‑5: cousin, eat, finish, nice, teacher
For a real demo, practice those six glosses (cousin, eat, finish, nice, student, teacher) plus the extra ones if you want to show top‑10 or top‑20. They match the order used in each run’s label_to_id.json, so the predictions you see in the app will use the same wording.

How2Sign top‑k gloss sets:

30000-cache / top‑5 runs: Good., Hi!, Okay., Okay?, There we go.
30000-cache / top‑10 run: Good., Hi!, Okay., Okay?, There we go., Hi., One, two, three, four., Alright., Lift., Step.
9000-cache / top‑5 runs (and how2sign_top5_seq60_bi): Hi!, There we go., Good., Okay., One and two and three and four.
20000-cache / top‑5 runs: Hi!, Good., Okay., There we go., Hi. (same set, minor duplication of “Hi!” vs. “Hi.”)
For the current demo we’re using the 9k/30k top‑5 snapshots, so practice these phrases:

“Hi!”
“There we go.”
“Good.”
“Okay.”
“One and two and three and four.” (rhythmic counting)
Those are the exact strings the model outputs. Use the exclamation/question punctuation when you label the UI, since the NLP stage distinguishes between “Hi!” and “Hi.”.

For the WLASL top‑6 model we trained (stored at artifacts/wlasl/top6_seq30_h128_l2_norm/), the glosses are:

before
candy
computer
drink
go![1773721403730](image/HerearetheglossesforeachMS-ASLtop/1773721403730.png)
who
Those are the exact labels the LSTM predicts (in that order), so practice those six signs for the demo.