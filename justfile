# https://just.systems/man/en/

set dotenv-load := true

PACKAGE := "agri"
REPOSITORY := "agri"
SOURCES := "src/"
TESTS := "tests/"

# display help information
default:
    @just --list

import 'tasks/mlflow.just'
import 'tasks/project.just'
import 'tasks/check.just'
import 'tasks/app.just'
import 'tasks/commit.just'
import 'tasks/install.just'
import 'tasks/clean.just'
import 'tasks/format.just'
import 'tasks/docker.just'
import 'tasks/docs.just'
