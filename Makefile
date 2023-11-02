code-archive:
	tar --no-xattrs -X .tarignore -czvf code_archive.tar.gz --totals . 

upload-code:
	scp code_archive.tar.gz yw580@login-gpu.hpc.cam.ac.uk:/home/yw580

data-archive:
	tar --no-xattrs -X .tarignore -czvf data_archive.tar.gz --totals -C /Users/yuval/GitHub/NEMO-Corpus/data .

upload-data:
	scp data_archive.tar.gz yw580@login-gpu.hpc.cam.ac.uk:/rds/user/yw580/hpc-work/ner-corpus

fasttext-archive:
	tar --no-xattrs -czvf fasttext_archive.tar.gz --totals fasttext/

upload-fasttext:
	scp fasttext_archive.tar.gz yw580@login-gpu.hpc.cam.ac.uk:/rds/user/yw580/hpc-work

clean:
	rm -rf *archive.tar.gz