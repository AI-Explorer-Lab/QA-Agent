pipeline {
  agent any

  stages {
    stage('Install') {
      steps {
        bat 'python -m pip install -r requirements.txt'
      }
    }

    stage('Test') {
      steps {
        bat 'pytest -q'
      }
    }

    stage('Build Image') {
      when {
        expression { fileExists('Dockerfile') }
      }
      steps {
        bat 'docker build -t trusted-qa-agent:latest .'
      }
    }
  }
}
