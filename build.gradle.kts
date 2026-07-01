plugins {
    kotlin("jvm") version "2.0.21"
    application
}

group = "ragindexer"
version = "1.0.0"

kotlin {
    jvmToolchain(20)
}

application {
    mainClass.set("ragindexer.MainKt")
}

dependencies {
    implementation("org.apache.pdfbox:pdfbox:3.0.3")
}

tasks.jar {
    archiveBaseName.set("rag-indexer")
    archiveVersion.set("")
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE

    manifest {
        attributes["Main-Class"] = application.mainClass.get()
    }

    from({
        configurations.runtimeClasspath.get()
            .filter { it.exists() }
            .map { if (it.isDirectory) it else zipTree(it) }
    })
}
