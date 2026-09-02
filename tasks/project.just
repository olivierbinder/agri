# run project tasks
[group('project')]
project: project-environment (project-run "tuning") (project-run "training") (project-run "evaluations") (project-run "promotion") (project-run "explanations") (project-run "inference")

# run tuning, training, and evaluations pipeline
[group('project')]
project-pipeline: (project-run "tuning") (project-run "training") (project-run "evaluations")

# export environment file
[group('project')]
project-environment:
    #!/usr/bin/env bash
    echo 'python: "3.12"' > python_env.yaml
    echo 'build_dependencies:' >> python_env.yaml
    echo '  - pip' >> python_env.yaml
    echo '  - setuptools' >> python_env.yaml
    echo '  - wheel' >> python_env.yaml
    echo 'dependencies:' >> python_env.yaml
    echo '  - .' >> python_env.yaml

# run project job using python directly in the uv environment
[group('project')]
project-run job:
    uv run python3 -m agri confs/{{job}}.yaml
