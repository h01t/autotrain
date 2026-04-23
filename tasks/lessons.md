# Lessons Learned

- Before cloning or recreating a repo the user mentions, search the local `~/Dev` area first and prefer the existing checkout if one is already present.
- When wiring AutoTrain for an external ML repo, inspect that repo's hardware/backend resolution logic and explicitly request GPU if the project defaults to CPU-safe `auto`.
