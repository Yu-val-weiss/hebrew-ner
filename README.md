# Hebrew NER: Part II Project

## Forked YAP Parser

### New Command

`yap/heb/joint/lattice` takes in a JSON object with the following structure

```JSON
{"amb_lattice": "0\t1\tב\tב\tPREPOSITION\tPREPOSITION\t_\t1\n0\t3\tבגן\tבגן\tNNP\tNNP\tgen=M|num=S\t1\n0\t3\tבגן\tבגן\tNN\tNN\tgen=M|num=P|num=S\t1\n0\t3\tבגן\tבגן\tNN\tNN\tgen=M|num=S\t1\n0\t3\tבגן\tבגן\tNNP\tNNP\tgen=F|num=S\t1\n0\t3\tבגן\tבגן\tNNP\tNNP\tgen=F|gen=M|num=S\t1\n0\t3\tבגן\tבגן\tNNP\tNNP\t_\t1\n0\t3\tבגן\tבגן\tNN\tNN\tgen=M|num=P\t1\n0\t3\tבגן\tבגן\tNN\tNN\tgen=F|num=S\t1\n0\t3\tבגן\tבגן\tNN\tNN\tgen=F|num=P\t1\n1\t3\tגן\tגן\tNN\tNN\tgen=M|num=S\t1\n1\t3\tגן\tגן\tNNT\tNNT\tgen=M|num=S\t1\n1\t2\tה\tה\tDEF\tDEF\t_\t1\n2\t3\tגן\tגן\tNNT\tNNT\tgen=M|num=S\t1\n2\t3\tגן\tגן\tNN\tNN\tgen=M|num=S\t1\n\n"}
```

#### Notes

- There **MUST** be two whitespace characters at the end
- The MA step returns it as `{"ma_lattice" : ""}`
- Go version must be 1.15

### Rebuilding

If a new version has been pushed to GitHub run the following:

```zsh
docker compose build --no-cache
```

```zsh
docker compose up
```
