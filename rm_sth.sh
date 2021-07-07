rm_exist() {
    [ -e $1 ] && rm $1
}
rm -rf ckps
rm -rf results
rm -rf checkpoints
rm_exist tb.log
rm_exist run.log
rm_exist chain.log
