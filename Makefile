code-archive:
	tar --no-xattrs -X .tarignore -czvf code_archive.tar.gz --totals . 

upload-code:
	rsync -av --progress --exclude-from='ignore.txt' . yw580@login.hpc.cam.ac.uk:/home/yw580/hebrew-ner

data-archive:
	tar --no-xattrs -X .tarignore -czvf data_archive.tar.gz --totals -C /Users/yuval/GitHub/NEMO-Corpus data/

upload-data:
	scp data_archive.tar.gz yw580@login.hpc.cam.ac.uk:/rds/user/yw580/hpc-work/ner-corpus

fasttext-archive:
	tar --no-xattrs -czvf fasttext_archive.tar.gz --totals fasttext/

upload-fasttext:
	scp fasttext_archive.tar.gz yw580@login.hpc.cam.ac.uk:/rds/user/yw580/hpc-work

upload-slurm:
	rsync -av --progress slurm yw580@login.hpc.cam.ac.uk:/home/yw580/hebrew-ner

upload-configs:
	rsync -av --progress configs yw580@login.hpc.cam.ac.uk:/home/yw580/hebrew-ner

upload-slurm-configs:
	rsync -av --progress configs slurm yw580@login.hpc.cam.ac.uk:/home/yw580/hebrew-ner

clean:
	rm -rf *archive.tar.gz

archive.tar.gz:
	git ls-files -z | xargs -0 tar -czvf archive.tar.gz                   

run-yap:
	cd /Users/yuval/go/src/yap && go build . && ./yap api