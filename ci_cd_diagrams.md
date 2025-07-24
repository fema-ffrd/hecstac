# CI/CD Pipeline

The CI/CD pipeline is managed with GitHub Actions and automates the build, test, and release processes.

### Push to `main` CI/CD Process Diagram

```mermaid
graph LR
    A[Push to main] --> B{Run Tests};
    B --> C{Get Version};
    C --> D{Check Version};
    D -->|Version Exists| E[Fail];
    D -->|New Version| F{Docker Build & Push};
    D -->|New Version| G{Publish to PyPI};
    D -->|New Version| H{Create GitHub Release};
    A --> I{Build and Publish Docs};
    F --> J[GitHub Container Registry];
    G --> K[PyPI];
    H --> L[GitHub Release];
    subgraph Main Branch
        A[Push to main];
        I[Build and Publish Docs];
    end
    style E fill:#f94144,color:#fff
    style J fill:#90be6d,color:#fff
    style K fill:#90be6d,color:#fff
    style L fill:#90be6d,color:#fff
    style B fill:#43aa8b,color:#fff
    style C fill:#43aa8b,color:#fff
    style D fill:#43aa8b,color:#fff
    style F fill:#43aa8b,color:#fff
    style G fill:#43aa8b,color:#fff
    style H fill:#43aa8b,color:#fff
    style I fill:#43aa8b,color:#fff
```

### Push to `dev` CI/CD Process Diagram

```mermaid
graph LR
    A[Push to dev] --> B{Run Tests};
    B --> C{Get Version};
    C --> D{Check Version};
    D -->|Version Exists| E[Warn];
    D --> F{Docker Build & Push};
    F --> G[GitHub Container Registry];
    subgraph Dev Branch
        A[Push to dev];
    end
    style E fill:#f94144,color:#fff
    style G fill:#90be6d,color:#fff
    style B fill:#43aa8b,color:#fff
    style C fill:#43aa8b,color:#fff
    style D fill:#43aa8b,color:#fff
    style F fill:#43aa8b,color:#fff
```

### Open PR to `dev`/`main` CI/CD Process Diagram

```mermaid
graph LR
    A[Open PR to dev/main] --> B{Run Tests};
    B --> C{Get Version};
    C --> D{Test Docker Build};
    style B fill:#43aa8b,color:#fff
    style C fill:#43aa8b,color:#fff
    style D fill:#43aa8b,color:#fff
```

### Docs

The `Read the Docs` build and publish step is triggered on push to the `main` branch and handled externally by `Read the Docs`.
