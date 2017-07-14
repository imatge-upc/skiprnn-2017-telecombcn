## Skipping State Updates in Recurrent Neural Networks

Remember to rename `gh-pages-in-progress` to `gh-pages`:

```
git branch -m old_branch new_branch         # Rename branch locally    
git push origin :old_branch                 # Delete the old branch    
git push --set-upstream origin new_branch   # Push the new branch, set local branch to track the new remote
```

To push to this branch:

```
git push origin gh-pages-in-progress
```
