# Contributing

This project is an experimental memory-system lab. Contributions should preserve
the core boundaries:

- evidence is not replaced by summaries;
- LLM output is candidate material unless explicitly reviewed;
- graph candidates and graph metrics are not proof;
- live provider calls must be opt-in;
- private data and generated workspaces stay out of the repository.

Before opening a change, run:

```powershell
python -m unittest discover -s tests
```

For non-mainline capabilities, prefer mature external research, resources, and
tools instead of hand-rolled toy implementations.
