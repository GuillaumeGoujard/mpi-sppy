the results.csv in this file should not be in the git repo (copy it
to xfer then add, commit, and push)

cp results.csv xfer/.
cd xfer
git add results.csv
git commit -m "transferring dark sky results"
git push
