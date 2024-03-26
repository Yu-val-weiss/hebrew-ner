.PHONY: *

code-archive:
	tar --no-xattrs -X .tarignore -czvf code_archive.tar.gz --totals . 

upload-code:
	rsync -av --progress --exclude-from='ignore.txt' . ${CRSID}@login.hpc.cam.ac.uk:/home/${CRSID}/hebrew-ner --human-readable

data-archive:
	tar --no-xattrs -X .tarignore -czvf data_archive.tar.gz --totals -C /Users/yuval/GitHub/NEMO-Corpus data/

upload-data:
	scp data_archive.tar.gz ${CRSID}@login.hpc.cam.ac.uk:/rds/user/${CRSID}/hpc-work/ner-corpus

fasttext-archive:
	tar --no-xattrs -czvf fasttext_archive.tar.gz --totals fasttext/

upload-fasttext:
	scp fasttext_archive.tar.gz ${CRSID}@login.hpc.cam.ac.uk:/rds/user/${CRSID}/hpc-work

upload-slurm:
	rsync -av --progress slurm ${CRSID}@login.hpc.cam.ac.uk:/home/${CRSID}/hebrew-ner --human-readable --delete

upload-configs:
	rsync -av --progress configs ${CRSID}@login.hpc.cam.ac.uk:/home/${CRSID}/hebrew-ner --human-readable --delete

upload-slurm-configs:
	rsync -av --progress configs slurm ${CRSID}@login.hpc.cam.ac.uk:/home/${CRSID}/hebrew-ner --human-readable --delete

clean:
	rm -rf *archive.tar.gz

archive:
	git ls-files -z | xargs -0 tar -czvf archive.tar.gz                   

run-yap:
	cd ${YAP_PATH} && go build . && ./yap api