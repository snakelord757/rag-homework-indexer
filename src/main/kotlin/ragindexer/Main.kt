package ragindexer

import org.apache.pdfbox.Loader
import org.apache.pdfbox.text.PDFTextStripper
import java.io.IOException
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.security.CodeSource
import java.time.Duration
import kotlin.io.path.extension
import kotlin.io.path.fileSize
import kotlin.io.path.isRegularFile
import kotlin.io.path.name

fun main(args: Array<String>) {
    if (args.any { it == "-h" || it == "--help" }) {
        Cli.printUsage()
        return
    }

    try {
        when (args.firstOrNull()) {
            "ask" -> runAsk(args.drop(1).toTypedArray())
            "chat" -> runChat(args.drop(1).toTypedArray())
            else -> runIndex(args)
        }
    } catch (ex: CliException) {
        System.err.println(ex.message)
        System.err.println()
        Cli.printUsage()
        kotlin.system.exitProcess(2)
    } catch (ex: Exception) {
        System.err.println("Error: ${ex.message}")
        kotlin.system.exitProcess(1)
    }
}

private fun runIndex(args: Array<String>) {
    val config = Cli.parseIndex(args)
    val workDir = Paths.get("").toAbsolutePath().normalize()
    val document = findDocument(workDir, config.documentName)

    println("Working directory: $workDir")
    println("Document: $document")
    println("Strategy: ${config.strategy}")
    println("Ollama: ${config.ollamaUrl}, model: ${config.model}")

    val extractor = DocumentExtractor()
    val extracted = extractor.extract(document, config.pdfPages)
    if (extracted.segments.none { it.text.isNotBlank() }) {
        error("Document text is empty after extraction.")
    }

    val ollama = OllamaEmbeddingClient(config.ollamaUrl, config.model)
    val chunks = when (config.strategy) {
        ChunkStrategy.FIXED -> FixedSizeChunker(config.fixedSize, config.overlap).chunk(extracted)
        ChunkStrategy.SEMANTIC -> SemanticChunker(
            ollama = ollama,
            maxChunkChars = config.semanticMaxChars,
            similarityThreshold = config.semanticSimilarityThreshold
        ).chunk(extracted)
    }.mapIndexed { index, chunk ->
        chunk.copy(metadata = chunk.metadata.copy(chunkId = "chunk-${index + 1}"))
    }

    println("Chunks prepared: ${chunks.size}")
    val embedded = chunks.mapIndexed { index, chunk ->
        println("Embedding ${index + 1}/${chunks.size}: ${chunk.metadata.chunkId} (${chunk.text.length} chars)")
        IndexedChunk(chunk.metadata, chunk.text, ollama.embed(chunk.text))
    }

    val output = jarDirectory().resolve("${document.fileNameWithoutExtension()}-index.json")
    val index = IndexFile(
        document = document.toAbsolutePath().normalize().toString(),
        model = config.model,
        strategy = config.strategy.cliName,
        chunks = embedded
    )
    Files.writeString(output, JsonWriter.write(index), StandardCharsets.UTF_8)
    println("Done. Index saved to: ${output.toAbsolutePath().normalize()}")
}

private fun runAsk(args: Array<String>) {
    val config = Cli.parseAsk(args)
    val indexPath = Paths.get(config.indexPath).toAbsolutePath().normalize()
    if (!Files.isRegularFile(indexPath)) error("Index file was not found: $indexPath")

    val index = IndexReader.read(indexPath)
    val embeddingModel = config.embeddingModel ?: index.model
    val ollama = OllamaEmbeddingClient(config.ollamaUrl, embeddingModel)

    println("Index: $indexPath")
    println("Embedding model: $embeddingModel")
    println("DeepSeek model: ${config.deepSeekModel}")

    val apiKey = resolveDeepSeekApiKey(config.deepSeekApiKey)
    val deepSeek = DeepSeekChatClient(config.deepSeekUrl, apiKey, config.deepSeekModel, config.temperature)

    val questions = config.autoFile?.let(::readQuestionsFile) ?: listOf(config.question)
    if (config.autoFile != null) println("Auto questions: ${questions.size}")

    questions.forEachIndexed { questionIndex, question ->
        if (questions.size > 1) {
            println()
            println("Question ${questionIndex + 1}/${questions.size}:")
            println(question)
        }
        answerWithRag(
            question = question,
            indexes = listOf(
                LoadedIndex(
                    path = indexPath,
                    index = index,
                    embeddingModel = embeddingModel,
                    ollama = ollama
                )
            ),
            deepSeek = deepSeek,
            searchTopK = config.searchTopK,
            topK = config.topK,
            relevanceThreshold = config.relevanceThreshold
        )
    }
}

private fun runChat(args: Array<String>) {
    val config = Cli.parseChat(args)
    val apiKey = resolveDeepSeekApiKey(config.deepSeekApiKey)

    println("DeepSeek model: ${config.deepSeekModel}")
    val deepSeek = DeepSeekChatClient(config.deepSeekUrl, apiKey, config.deepSeekModel, config.temperature)
    val questions = config.autoFile?.let(::readQuestionsFile) ?: listOf(config.question)
    if (config.autoFile != null) println("Auto questions: ${questions.size}")

    if (config.ragEnabled) {
        val indexes = loadRagIndexes(config)

        println("RAG: enabled")
        println("Indexes: ${indexes.size}")
        indexes.forEach { loaded ->
            println("- ${loaded.path}: model=${loaded.embeddingModel}, chunks=${loaded.index.chunks.size}")
        }
        println()

        questions.forEachIndexed { questionIndex, question ->
            if (questions.size > 1) {
                println()
                println("Question ${questionIndex + 1}/${questions.size}:")
                println(question)
            }
            answerWithRag(
                question = question,
                indexes = indexes,
                deepSeek = deepSeek,
                searchTopK = config.searchTopK,
                topK = config.topK,
                relevanceThreshold = config.relevanceThreshold
            )
        }
        return
    }

    println("RAG: disabled")
    println()

    questions.forEachIndexed { questionIndex, question ->
        if (questions.size > 1) {
            println()
            println("Question ${questionIndex + 1}/${questions.size}:")
            println(question)
        }
        val answer = deepSeek.chat(question)
        println("Answer:")
        println(answer.toConsoleSafeText())
    }
}

private fun loadRagIndexes(config: ChatConfig): List<LoadedIndex> {
    val paths = if (config.indexPath != null) {
        listOf(Paths.get(config.indexPath).toAbsolutePath().normalize())
    } else {
        discoverIndexFiles()
    }
    if (paths.isEmpty()) error("No *-index.json files were found next to the jar or in the current directory.")

    return paths.map { path ->
        if (!Files.isRegularFile(path)) error("Index file was not found: $path")
        val index = IndexReader.read(path)
        val embeddingModel = config.embeddingModel ?: index.model
        LoadedIndex(
            path = path,
            index = index,
            embeddingModel = embeddingModel,
            ollama = OllamaEmbeddingClient(config.ollamaUrl, embeddingModel)
        )
    }
}

private fun discoverIndexFiles(): List<Path> {
    val roots = listOf(jarDirectory(), Paths.get("").toAbsolutePath().normalize()).distinct()
    return roots
        .flatMap { root ->
            if (!Files.isDirectory(root)) {
                emptyList()
            } else {
                Files.list(root).use { stream ->
                    stream
                        .filter { Files.isRegularFile(it) }
                        .filter { it.fileName.toString().endsWith("-index.json", ignoreCase = true) }
                        .map { it.toAbsolutePath().normalize() }
                        .toList()
                }
            }
        }
        .distinct()
        .sorted()
}

private fun answerWithRag(
    question: String,
    indexes: List<LoadedIndex>,
    deepSeek: DeepSeekChatClient,
    searchTopK: Int,
    topK: Int,
    relevanceThreshold: Double
) {
    println("Searching relevant chunks...")
    println("Retrieval settings: searchTopK=$searchTopK, finalTopK=$topK, rerankThreshold=${"%.4f".format(relevanceThreshold)}")

    if (indexes.all { it.index.chunks.isEmpty() }) {
        println("Answer:")
        println("Не знаю")
        return
    }

    val retrieved = searchRelevantChunks(question, indexes, searchTopK)
    if (retrieved.isEmpty()) {
        println("Answer:")
        println("Не знаю")
        return
    }

    println("Top chunks before DeepSeek reranker:")
    retrieved.forEachIndexed { index, result ->
        println("${index + 1}. ${result.chunk.metadata.chunkId}: embedding=${"%.4f".format(result.score)}")
    }

    val results = deepSeek.rerank(question, retrieved, relevanceThreshold, topK)
    println("Top chunks after DeepSeek reranker:")
    results.forEachIndexed { index, result ->
        println("${index + 1}. ${result.chunk.metadata.chunkId}: rerank=${"%.4f".format(result.score)}")
    }
    println()

    if (results.isEmpty()) {
        println("Answer:")
        println("Не знаю")
        return
    }

    val answer = deepSeek.answer(question, results)
    println("Answer:")
    println(answer.toConsoleSafeText())
}

private fun resolveDeepSeekApiKey(cliApiKey: String?): String {
    val apiKey = cliApiKey
        ?: System.getenv("DEEPSEEK_API_KEY")
        ?: System.getenv("DEEPSEEK_KEY_API")
        ?: readLocalProperty("DEEPSEEK_KEY_API")
    if (apiKey.isNullOrBlank()) {
        throw CliException(
            "DeepSeek API key is required. Pass --deepseek-api-key, set DEEPSEEK_API_KEY/DEEPSEEK_KEY_API, " +
                "or add DEEPSEEK_KEY_API to local.properties next to the jar."
        )
    }
    return apiKey
}

private fun findDocument(workDir: Path, documentName: String): Path {
    val exactPath = workDir.resolve(documentName).normalize()
    if (Files.isRegularFile(exactPath)) return exactPath

    Files.walk(workDir).use { stream ->
        val matches = stream
            .filter { it.isRegularFile() && it.fileName.toString().equals(documentName, ignoreCase = true) }
            .sorted()
            .toList()

        return when {
            matches.isEmpty() -> error("Document '$documentName' was not found under $workDir")
            matches.size == 1 -> matches.first()
            else -> error("Document name is ambiguous. Matches:\n${matches.joinToString("\n")}")
        }
    }
}

private fun jarDirectory(): Path {
    val source: CodeSource = object {}.javaClass.protectionDomain.codeSource
    val location = Paths.get(source.location.toURI()).toAbsolutePath().normalize()
    return if (Files.isRegularFile(location)) location.parent else location
}

private fun readLocalProperty(key: String): String? {
    val localProperties = jarDirectory().resolve("local.properties")
    if (!Files.isRegularFile(localProperties)) return null

    return Files.readAllLines(localProperties, StandardCharsets.UTF_8)
        .asSequence()
        .map { it.trim() }
        .filter { it.isNotEmpty() && !it.startsWith("#") && !it.startsWith("!") }
        .mapNotNull { line ->
            val separator = line.indexOf('=').takeIf { it >= 0 } ?: line.indexOf(':').takeIf { it >= 0 }
            if (separator == null) null else line.substring(0, separator).trim() to line.substring(separator + 1).trim()
        }
        .firstOrNull { (propertyKey, value) -> propertyKey == key && value.isNotBlank() }
        ?.second
}

private fun readQuestionsFile(pathText: String): List<String> {
    val path = Paths.get(pathText).toAbsolutePath().normalize()
    if (!Files.isRegularFile(path)) error("Questions file was not found: $path")

    val questions = Files.readAllLines(path, StandardCharsets.UTF_8)
        .map { it.trim() }
        .filter { it.isNotEmpty() && !it.startsWith("#") }

    if (questions.isEmpty()) error("Questions file does not contain questions: $path")
    return questions
}

private fun String.toConsoleSafeText(): String =
    this
        .replace('\u2018', '\'')
        .replace('\u2019', '\'')
        .replace('\u201A', '\'')
        .replace('\u201B', '\'')
        .replace('\u201C', '"')
        .replace('\u201D', '"')
        .replace('\u201E', '"')
        .replace('\u201F', '"')
        .replace('\u00AB', '"')
        .replace('\u00BB', '"')
        .replace('\u2013', '-')
        .replace('\u2014', '-')
        .replace('\u2212', '-')
        .replace("\u2026", "...")

private fun Path.fileNameWithoutExtension(): String {
    val file = fileName.toString()
    val dot = file.lastIndexOf('.')
    return if (dot > 0) file.substring(0, dot) else file
}

enum class ChunkStrategy(val cliName: String) {
    FIXED("fixed"),
    SEMANTIC("semantic");

    companion object {
        fun fromCli(value: String): ChunkStrategy = entries.firstOrNull { it.cliName == value.lowercase() }
            ?: throw CliException("Unknown strategy '$value'. Expected: fixed or semantic.")
    }
}

data class CliConfig(
    val documentName: String,
    val strategy: ChunkStrategy,
    val fixedSize: Int,
    val overlap: Int,
    val semanticMaxChars: Int,
    val semanticSimilarityThreshold: Double,
    val pdfPages: Int?,
    val model: String,
    val ollamaUrl: String
)

data class AskConfig(
    val indexPath: String,
    val question: String,
    val autoFile: String?,
    val searchTopK: Int,
    val topK: Int,
    val relevanceThreshold: Double,
    val embeddingModel: String?,
    val ollamaUrl: String,
    val deepSeekModel: String,
    val deepSeekUrl: String,
    val deepSeekApiKey: String?,
    val temperature: Double
)

data class ChatConfig(
    val ragEnabled: Boolean,
    val indexPath: String?,
    val question: String,
    val autoFile: String?,
    val searchTopK: Int,
    val topK: Int,
    val relevanceThreshold: Double,
    val embeddingModel: String?,
    val ollamaUrl: String,
    val deepSeekModel: String,
    val deepSeekUrl: String,
    val deepSeekApiKey: String?,
    val temperature: Double
)

data class LoadedIndex(
    val path: Path,
    val index: IndexFile,
    val embeddingModel: String,
    val ollama: OllamaEmbeddingClient
)

class CliException(message: String) : RuntimeException(message)

object Cli {
    fun parseIndex(args: Array<String>): CliConfig {
        if (args.isEmpty()) {
            throw CliException("Missing required arguments.")
        }

        var document: String? = null
        var strategy = ChunkStrategy.FIXED
        var fixedSize = 1200
        var overlap = 150
        var semanticMaxChars = 1600
        var semanticThreshold = 0.72
        var pages: Int? = null
        var model = "qwen3-embedding"
        var ollamaUrl = "http://localhost:11434"

        var i = 0
        while (i < args.size) {
            when (val arg = args[i]) {
                "--strategy" -> strategy = ChunkStrategy.fromCli(args.valueAfter(i++, arg))
                "--size" -> fixedSize = args.valueAfter(i++, arg).positiveInt(arg)
                "--overlap" -> overlap = args.valueAfter(i++, arg).nonNegativeInt(arg)
                "--max-chars" -> semanticMaxChars = args.valueAfter(i++, arg).positiveInt(arg)
                "--threshold" -> semanticThreshold = args.valueAfter(i++, arg).toDoubleOrNull()
                    ?: throw CliException("$arg must be a number.")
                "--pages" -> pages = args.valueAfter(i++, arg).positiveInt(arg)
                "--model" -> model = args.valueAfter(i++, arg)
                "--ollama-url" -> ollamaUrl = args.valueAfter(i++, arg).trimEnd('/')
                else -> {
                    if (arg.startsWith("--")) throw CliException("Unknown option '$arg'.")
                    if (document != null) throw CliException("Only one document name can be provided.")
                    document = arg
                }
            }
            i++
        }

        if (document.isNullOrBlank()) throw CliException("Document name is required.")
        if (overlap >= fixedSize) throw CliException("--overlap must be smaller than --size.")
        if (semanticThreshold !in -1.0..1.0) throw CliException("--threshold must be between -1.0 and 1.0.")

        return CliConfig(
            documentName = document,
            strategy = strategy,
            fixedSize = fixedSize,
            overlap = overlap,
            semanticMaxChars = semanticMaxChars,
            semanticSimilarityThreshold = semanticThreshold,
            pdfPages = pages,
            model = model,
            ollamaUrl = ollamaUrl
        )
    }

    fun parseAsk(args: Array<String>): AskConfig {
        if (args.size < 2) {
            throw CliException("Missing ask arguments.")
        }

        val positional = mutableListOf<String>()
        var topK = 5
        var searchTopK: Int? = null
        var relevanceThreshold = 0.6
        var embeddingModel: String? = null
        var ollamaUrl = "http://localhost:11434"
        var deepSeekModel = "deepseek-v4-flash"
        var deepSeekUrl = "https://api.deepseek.com"
        var deepSeekApiKey: String? = null
        var temperature = 0.2

        var i = 0
        while (i < args.size) {
            when (val arg = args[i]) {
                "--top-k" -> topK = args.valueAfter(i++, arg).positiveInt(arg)
                "--search-top-k" -> searchTopK = args.valueAfter(i++, arg).positiveInt(arg)
                "--rerank-threshold", "--similarity-threshold" -> relevanceThreshold = args.valueAfter(i++, arg).toDoubleOrNull()
                    ?: throw CliException("$arg must be a number.")
                "--embed-model" -> embeddingModel = args.valueAfter(i++, arg)
                "--ollama-url" -> ollamaUrl = args.valueAfter(i++, arg).trimEnd('/')
                "--deepseek-model" -> deepSeekModel = args.valueAfter(i++, arg)
                "--deepseek-url" -> deepSeekUrl = args.valueAfter(i++, arg).trimEnd('/')
                "--deepseek-api-key" -> deepSeekApiKey = args.valueAfter(i++, arg)
                "--temperature" -> temperature = args.valueAfter(i++, arg).toDoubleOrNull()
                    ?: throw CliException("$arg must be a number.")
                else -> {
                    if (arg.startsWith("--")) throw CliException("Unknown option '$arg'.")
                    positional += arg
                }
            }
            i++
        }

        if (positional.size < 2) {
            throw CliException("Usage for ask mode requires <index-json> and <question>, or <index-json> auto <file>.")
        }
        if (temperature !in 0.0..2.0) throw CliException("--temperature must be between 0 and 2.")
        if (relevanceThreshold !in 0.0..1.0) throw CliException("--rerank-threshold must be between 0.0 and 1.0.")
        val effectiveSearchTopK = searchTopK ?: topK * 3
        if (effectiveSearchTopK < topK) throw CliException("--search-top-k must be greater than or equal to --top-k.")
        val autoFile = if (positional.getOrNull(1) == "auto") {
            if (positional.size != 3) throw CliException("Usage for ask autopilot: ask <index-json> auto <questions-file>.")
            positional[2]
        } else {
            null
        }

        return AskConfig(
            indexPath = positional.first(),
            question = if (autoFile == null) positional.drop(1).joinToString(" ") else "",
            autoFile = autoFile,
            searchTopK = effectiveSearchTopK,
            topK = topK,
            relevanceThreshold = relevanceThreshold,
            embeddingModel = embeddingModel,
            ollamaUrl = ollamaUrl,
            deepSeekModel = deepSeekModel,
            deepSeekUrl = deepSeekUrl,
            deepSeekApiKey = deepSeekApiKey,
            temperature = temperature
        )
    }

    fun parseChat(args: Array<String>): ChatConfig {
        if (args.isEmpty()) {
            throw CliException("Missing chat question.")
        }

        val positional = mutableListOf<String>()
        var ragEnabled = true
        var topK = 5
        var searchTopK: Int? = null
        var relevanceThreshold = 0.6
        var embeddingModel: String? = null
        var ollamaUrl = "http://localhost:11434"
        var deepSeekModel = "deepseek-v4-flash"
        var deepSeekUrl = "https://api.deepseek.com"
        var deepSeekApiKey: String? = null
        var temperature = 0.2

        var i = 0
        while (i < args.size) {
            when (val arg = args[i]) {
                "--no-rag" -> ragEnabled = false
                "--top-k" -> topK = args.valueAfter(i++, arg).positiveInt(arg)
                "--search-top-k" -> searchTopK = args.valueAfter(i++, arg).positiveInt(arg)
                "--rerank-threshold", "--similarity-threshold" -> relevanceThreshold = args.valueAfter(i++, arg).toDoubleOrNull()
                    ?: throw CliException("$arg must be a number.")
                "--embed-model" -> embeddingModel = args.valueAfter(i++, arg)
                "--ollama-url" -> ollamaUrl = args.valueAfter(i++, arg).trimEnd('/')
                "--deepseek-model" -> deepSeekModel = args.valueAfter(i++, arg)
                "--deepseek-url" -> deepSeekUrl = args.valueAfter(i++, arg).trimEnd('/')
                "--deepseek-api-key" -> deepSeekApiKey = args.valueAfter(i++, arg)
                "--temperature" -> temperature = args.valueAfter(i++, arg).toDoubleOrNull()
                    ?: throw CliException("$arg must be a number.")
                else -> {
                    if (arg.startsWith("--")) throw CliException("Unknown option '$arg'.")
                    positional += arg
                }
            }
            i++
        }

        if (temperature !in 0.0..2.0) throw CliException("--temperature must be between 0 and 2.")
        if (relevanceThreshold !in 0.0..1.0) throw CliException("--rerank-threshold must be between 0.0 and 1.0.")
        val effectiveSearchTopK = searchTopK ?: topK * 3
        if (effectiveSearchTopK < topK) throw CliException("--search-top-k must be greater than or equal to --top-k.")

        val indexPath = if (ragEnabled && positional.firstOrNull()?.endsWith("-index.json", ignoreCase = true) == true) {
            positional.first()
        } else {
            null
        }
        val questionArgs = if (indexPath != null) positional.drop(1) else positional
        if (questionArgs.isEmpty()) throw CliException("Usage for chat mode requires <question>, or auto <questions-file>.")
        val autoFile = if (questionArgs.firstOrNull() == "auto") {
            if (questionArgs.size != 2) throw CliException(
                if (ragEnabled) {
                    "Usage for chat autopilot: chat [<index-json>] auto <questions-file>."
                } else {
                    "Usage for chat without RAG autopilot: chat --no-rag auto <questions-file>."
                }
            )
            questionArgs[1]
        } else {
            null
        }

        return ChatConfig(
            ragEnabled = ragEnabled,
            indexPath = indexPath,
            question = if (autoFile == null) questionArgs.joinToString(" ") else "",
            autoFile = autoFile,
            searchTopK = effectiveSearchTopK,
            topK = topK,
            relevanceThreshold = relevanceThreshold,
            embeddingModel = embeddingModel,
            ollamaUrl = ollamaUrl,
            deepSeekModel = deepSeekModel,
            deepSeekUrl = deepSeekUrl,
            deepSeekApiKey = deepSeekApiKey,
            temperature = temperature
        )
    }

    fun printUsage() {
        println(
            """
            Usage:
              java -jar rag-indexer.jar <document-name> --strategy fixed [--size 1200] [--overlap 150] [--pages N]
              java -jar rag-indexer.jar <document-name> --strategy semantic [--max-chars 1600] [--threshold 0.72] [--pages N]
              java -jar rag-indexer.jar ask <index-json> "<question>" [--search-top-k 15] [--top-k 5] [--rerank-threshold 0.6] [--embed-model MODEL]
              java -jar rag-indexer.jar ask <index-json> auto <questions-file> [--search-top-k 15] [--top-k 5] [--rerank-threshold 0.6]
              java -jar rag-indexer.jar chat "<question>"
              java -jar rag-indexer.jar chat auto <questions-file>
              java -jar rag-indexer.jar chat <index-json> "<question>"
              java -jar rag-indexer.jar chat <index-json> auto <questions-file>
              java -jar rag-indexer.jar chat --no-rag "<question>"
              java -jar rag-indexer.jar chat --no-rag auto <questions-file>

            Options:
              --model MODEL          Ollama embedding model, default: qwen3-embedding
              --ollama-url URL       Ollama base URL, default: http://localhost:11434

            Ask options:
              --search-top-k N          Chunks to retrieve before DeepSeek reranking, default: --top-k * 3
              --top-k N                 Chunks to keep after DeepSeek reranking, default: 5
              --rerank-threshold N      Minimum DeepSeek relevance score, default: 0.6
              --embed-model MODEL        Ollama embedding model for the question, default: model from index
              --deepseek-model MODEL     DeepSeek model, default: deepseek-v4-flash
              --deepseek-url URL         DeepSeek base URL, default: https://api.deepseek.com
              --deepseek-api-key KEY     DeepSeek API key, default: env variable or local.properties
              --temperature NUMBER       DeepSeek temperature, default: 0.2

            Chat mode uses all discovered *-index.json files by default.
            Index discovery checks the jar directory and the current console directory.
            Use --no-rag before the question to send it directly to DeepSeek.
            Auto mode reads one question per line; empty lines and lines starting with # are skipped.

            The current console directory is used as the search root for the document.
            The resulting JSON index is written next to the jar file.
            """.trimIndent()
        )
    }

    private fun Array<String>.valueAfter(index: Int, option: String): String =
        getOrNull(index + 1) ?: throw CliException("$option requires a value.")

    private fun String.positiveInt(option: String): Int =
        toIntOrNull()?.takeIf { it > 0 } ?: throw CliException("$option must be a positive integer.")

    private fun String.nonNegativeInt(option: String): Int =
        toIntOrNull()?.takeIf { it >= 0 } ?: throw CliException("$option must be a non-negative integer.")
}

data class ExtractedDocument(
    val source: Path,
    val title: String,
    val segments: List<DocumentSegment>
)

data class DocumentSegment(
    val text: String,
    val section: String?
)

class DocumentExtractor {
    fun extract(path: Path, pdfPages: Int?): ExtractedDocument {
        val extension = path.extension.lowercase()
        val title = path.fileName.toString()
        val segments = when (extension) {
            "pdf" -> extractPdf(path, pdfPages)
            "md", "markdown" -> extractMarkdown(path)
            else -> listOf(DocumentSegment(readText(path), null))
        }
        return ExtractedDocument(path.toAbsolutePath().normalize(), title, segments)
    }

    private fun extractPdf(path: Path, pages: Int?): List<DocumentSegment> {
        Loader.loadPDF(path.toFile()).use { document ->
            val limit = pages?.coerceAtMost(document.numberOfPages) ?: document.numberOfPages
            val stripper = PDFTextStripper()
            return (1..limit).map { page ->
                stripper.startPage = page
                stripper.endPage = page
                DocumentSegment(stripper.getText(document).trim(), "page $page")
            }.filter { it.text.isNotBlank() }
        }
    }

    private fun extractMarkdown(path: Path): List<DocumentSegment> {
        val segments = mutableListOf<DocumentSegment>()
        var section: String? = null
        val buffer = StringBuilder()

        Files.readAllLines(path, StandardCharsets.UTF_8).forEach { line ->
            val heading = Regex("""^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$""").matchEntire(line)
            if (heading != null) {
                flushMarkdownSegment(buffer, section, segments)
                section = heading.groupValues[1].trim()
            } else {
                buffer.appendLine(line)
            }
        }
        flushMarkdownSegment(buffer, section, segments)
        return segments.ifEmpty { listOf(DocumentSegment(readText(path), null)) }
    }

    private fun flushMarkdownSegment(
        buffer: StringBuilder,
        section: String?,
        segments: MutableList<DocumentSegment>
    ) {
        val text = buffer.toString().trim()
        if (text.isNotBlank()) segments += DocumentSegment(text, section)
        buffer.clear()
    }

    private fun readText(path: Path): String {
        if (path.fileSize() > 50L * 1024L * 1024L) {
            error("Refusing to read files larger than 50 MB as plain text: $path")
        }
        return try {
            Files.readString(path, StandardCharsets.UTF_8)
        } catch (ex: IOException) {
            Files.readString(path, Charsets.ISO_8859_1)
        }
    }
}

data class ChunkMetadata(
    val source: String,
    val title: String,
    val section: String?,
    val chunkId: String
)

data class Chunk(
    val metadata: ChunkMetadata,
    val text: String
)

class FixedSizeChunker(
    private val chunkSize: Int,
    private val overlap: Int
) {
    fun chunk(document: ExtractedDocument): List<Chunk> {
        val result = mutableListOf<Chunk>()
        document.segments.forEach { segment ->
            val normalized = normalizeWhitespace(segment.text)
            var start = 0
            while (start < normalized.length) {
                var end = (start + chunkSize).coerceAtMost(normalized.length)
                if (end < normalized.length) {
                    val lastSpace = normalized.lastIndexOf(' ', startIndex = end)
                    if (lastSpace > start + chunkSize / 2) end = lastSpace
                }
                val text = normalized.substring(start, end).trim()
                if (text.isNotBlank()) result += document.chunk(text, segment.section)
                if (end == normalized.length) break
                start = (end - overlap).coerceAtLeast(start + 1)
            }
        }
        return result
    }
}

class SemanticChunker(
    private val ollama: OllamaEmbeddingClient,
    private val maxChunkChars: Int,
    private val similarityThreshold: Double
) {
    fun chunk(document: ExtractedDocument): List<Chunk> {
        val units = document.segments.flatMap { segment ->
            splitSemanticUnits(segment.text).map { DocumentSegment(it, segment.section) }
        }.filter { it.text.isNotBlank() }

        if (units.isEmpty()) return emptyList()

        println("Semantic chunking: embedding ${units.size} text units")
        val unitVectors = units.mapIndexed { index, unit ->
            println("Semantic unit ${index + 1}/${units.size}")
            ollama.embed(unit.text)
        }

        val chunks = mutableListOf<Chunk>()
        val current = StringBuilder(units.first().text)
        var currentSection = units.first().section
        var previousVector = unitVectors.first()

        for (i in 1 until units.size) {
            val next = units[i]
            val similarity = cosine(previousVector, unitVectors[i])
            val wouldBeTooLong = current.length + 2 + next.text.length > maxChunkChars
            val sectionChanged = currentSection != next.section

            if (wouldBeTooLong || sectionChanged || similarity < similarityThreshold) {
                chunks += document.chunk(current.toString(), currentSection)
                current.clear()
                current.append(next.text)
                currentSection = next.section
            } else {
                current.append("\n\n").append(next.text)
            }
            previousVector = unitVectors[i]
        }

        if (current.isNotBlank()) chunks += document.chunk(current.toString(), currentSection)
        return chunks
    }

    private fun splitSemanticUnits(text: String): List<String> {
        val paragraphs = text
            .split(Regex("""\n\s*\n"""))
            .map { normalizeWhitespace(it) }
            .filter { it.isNotBlank() }

        return paragraphs.flatMap { paragraph ->
            if (paragraph.length <= maxChunkChars) {
                listOf(paragraph)
            } else {
                paragraph.split(Regex("""(?<=[.!?])\s+"""))
                    .fold(mutableListOf<StringBuilder>()) { acc, sentence ->
                        if (acc.isEmpty() || acc.last().length + sentence.length + 1 > maxChunkChars) {
                            acc += StringBuilder(sentence)
                        } else {
                            acc.last().append(' ').append(sentence)
                        }
                        acc
                    }
                    .map { it.toString() }
            }
        }
    }
}

private fun ExtractedDocument.chunk(text: String, section: String?): Chunk =
    Chunk(
        metadata = ChunkMetadata(
            source = source.toString(),
            title = title,
            section = section,
            chunkId = ""
        ),
        text = normalizeWhitespace(text)
    )

private fun normalizeWhitespace(text: String): String =
    text.replace(Regex("""[ \t\r\n]+"""), " ").trim()

private fun cosine(a: List<Double>, b: List<Double>): Double {
    val size = minOf(a.size, b.size)
    if (size == 0) return 0.0
    var dot = 0.0
    var normA = 0.0
    var normB = 0.0
    for (i in 0 until size) {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    return if (normA == 0.0 || normB == 0.0) 0.0 else dot / kotlin.math.sqrt(normA * normB)
}

class OllamaEmbeddingClient(
    ollamaUrl: String,
    private val model: String
) {
    private val endpoint = URI.create("${ollamaUrl.trimEnd('/')}/api/embeddings")
    private val client = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(10))
        .build()

    fun embed(text: String): List<Double> {
        val body = """{"model":"${JsonWriter.escape(model)}","prompt":"${JsonWriter.escape(text)}"}"""
        val request = HttpRequest.newBuilder(endpoint)
            .timeout(Duration.ofMinutes(3))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
            .build()

        val response = client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
        if (response.statusCode() !in 200..299) {
            error("Ollama returned HTTP ${response.statusCode()}: ${response.body()}")
        }
        return JsonWriter.readEmbedding(response.body())
    }
}

data class IndexedChunk(
    val metadata: ChunkMetadata,
    val text: String,
    val embedding: List<Double>
)

data class IndexFile(
    val document: String,
    val model: String,
    val strategy: String,
    val chunks: List<IndexedChunk>
)

data class SearchResult(
    val chunk: IndexedChunk,
    val score: Double
)

private fun searchRelevantChunks(
    question: String,
    index: IndexFile,
    ollama: OllamaEmbeddingClient,
    topK: Int
): List<SearchResult> {
    val questionEmbedding = ollama.embed(question)
    return index.chunks
        .map { chunk -> SearchResult(chunk, cosine(questionEmbedding, chunk.embedding)) }
        .sortedByDescending { it.score }
        .take(topK)
}

private fun searchRelevantChunks(
    question: String,
    indexes: List<LoadedIndex>,
    topK: Int
): List<SearchResult> =
    indexes
        .flatMap { loaded ->
            val questionEmbedding = loaded.ollama.embed(question)
            loaded.index.chunks.map { chunk ->
                SearchResult(chunk, cosine(questionEmbedding, chunk.embedding))
            }
        }
        .sortedByDescending { it.score }
        .take(topK)

class DeepSeekChatClient(
    deepSeekUrl: String,
    private val apiKey: String,
    private val model: String,
    private val temperature: Double
) {
    private val endpoint = URI.create("${deepSeekUrl.trimEnd('/')}/chat/completions")
    private val client = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(10))
        .build()

    fun chat(question: String): String {
        val systemPrompt = """
            You are a helpful assistant. Answer in Russian unless the user asks for another language.
        """.trimIndent()
        return send(systemPrompt, question)
    }

    fun rerank(
        question: String,
        results: List<SearchResult>,
        relevanceThreshold: Double,
        topK: Int
    ): List<SearchResult> {
        val systemPrompt = """
            You are a relevance reranker for a RAG system.
            Score each candidate chunk from 0.0 to 1.0 by how useful it is for answering the question.
            Use only these scores: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.
            Use the same score for the same question and chunk text across runs.
            Return only valid JSON with this shape: {"scores":[{"chunk_id":"chunk-1","score":0.92}]}.
            Include every provided chunk id exactly once. Do not add explanations.
        """.trimIndent()
        val response = send(systemPrompt, buildRerankPrompt(question, results), RERANK_TEMPERATURE)
        val scores = readRerankScores(response)
        val embeddingScores = results.associate { it.chunk.metadata.chunkId to it.score }

        return results
            .map { result ->
                SearchResult(result.chunk, scores[result.chunk.metadata.chunkId] ?: 0.0)
            }
            .sortedWith(
                compareByDescending<SearchResult> { it.score }
                    .thenByDescending { embeddingScores[it.chunk.metadata.chunkId] ?: 0.0 }
                    .thenBy { it.chunk.metadata.chunkId }
            )
            .filter { it.score >= relevanceThreshold }
            .take(topK)
    }

    fun answer(question: String, results: List<SearchResult>): String {
        val systemPrompt = """
            You are a RAG assistant. Answer in Russian using only the provided context.
            If the provided context is missing, irrelevant, ambiguous, or not enough to answer confidently, answer exactly: Не знаю
            Do not add explanations when answering Не знаю.
            Every factual statement in a non-empty answer must be supported by a direct quote from the context.
            Always include the source chunk id next to each quote, for example: "quoted text" (chunk-3).
            Do not answer from general knowledge. If you cannot quote a relevant chunk, answer exactly: Не знаю
            Use this format:
            Ответ: <short answer with inline quotes and chunk ids>
            Цитаты:
            - "..." (chunk-id)
        """.trimIndent()
        val userPrompt = buildUserPrompt(question, results)
        return send(systemPrompt, userPrompt)
    }

    private fun buildRerankPrompt(question: String, results: List<SearchResult>): String = buildString {
        appendLine("Question:")
        appendLine(question)
        appendLine()
        appendLine("Candidate chunks:")
        results.forEachIndexed { index, result ->
            val metadata = result.chunk.metadata
            appendLine()
            appendLine("[${index + 1}] chunk_id=${metadata.chunkId}, embedding_score=${"%.4f".format(result.score)}")
            appendLine("Title: ${metadata.title}")
            if (metadata.section != null) appendLine("Section: ${metadata.section}")
            appendLine("Text:")
            appendLine(result.chunk.text.take(RERANK_TEXT_LIMIT))
        }
    }

    private fun readRerankScores(content: String): Map<String, Double> {
        val root = JsonParser(extractJsonObject(content)).parseObject()
        return root.list("scores").associate { value ->
            val item = value.asObject("scores item")
            val chunkId = item.string("chunk_id")
            val score = (item["score"] as? Number)?.toDouble()
                ?: error("DeepSeek reranker score for $chunkId must be a number.")
            chunkId to score.coerceIn(0.0, 1.0)
        }
    }

    private fun extractJsonObject(content: String): String {
        val start = content.indexOf('{')
        if (start < 0) error("DeepSeek reranker did not return a JSON object: $content")

        var depth = 0
        var inString = false
        var escaped = false
        for (i in start until content.length) {
            val ch = content[i]
            if (inString) {
                when {
                    escaped -> escaped = false
                    ch == '\\' -> escaped = true
                    ch == '"' -> inString = false
                }
            } else {
                when (ch) {
                    '"' -> inString = true
                    '{' -> depth++
                    '}' -> {
                        depth--
                        if (depth == 0) return content.substring(start, i + 1)
                    }
                }
            }
        }
        error("DeepSeek reranker returned malformed JSON: $content")
    }

    private companion object {
        const val RERANK_TEXT_LIMIT = 1800
        const val RERANK_TEMPERATURE = 0.0
    }

    private fun send(systemPrompt: String, userPrompt: String, requestTemperature: Double = temperature): String {
        val body = buildRequestBody(systemPrompt, userPrompt, requestTemperature)

        val request = HttpRequest.newBuilder(endpoint)
            .timeout(Duration.ofMinutes(3))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer $apiKey")
            .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
            .build()

        val response = client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
        if (response.statusCode() !in 200..299) {
            error("DeepSeek returned HTTP ${response.statusCode()}: ${response.body()}")
        }
        return readChatContent(response.body())
    }

    private fun buildUserPrompt(question: String, results: List<SearchResult>): String = buildString {
        appendLine("Context chunks:")
        results.forEachIndexed { index, result ->
            val metadata = result.chunk.metadata
            appendLine()
            appendLine("[${index + 1}] chunk_id=${metadata.chunkId}, score=${"%.4f".format(result.score)}")
            appendLine("Title: ${metadata.title}")
            if (metadata.section != null) appendLine("Section: ${metadata.section}")
            appendLine("Text:")
            appendLine(result.chunk.text)
        }
        appendLine()
        appendLine("Question:")
        appendLine(question)
        appendLine()
        appendLine("Remember: answer must include direct quotes and chunk ids. If there is no quotable evidence, answer exactly: Не знаю")
    }

    private fun buildRequestBody(systemPrompt: String, userPrompt: String, requestTemperature: Double): String = buildString {
        appendLine("{")
        appendLine("  \"model\": \"${JsonWriter.escape(model)}\",")
        appendLine("  \"messages\": [")
        appendLine("    {\"role\": \"system\", \"content\": \"${JsonWriter.escape(systemPrompt)}\"},")
        appendLine("    {\"role\": \"user\", \"content\": \"${JsonWriter.escape(userPrompt)}\"}")
        appendLine("  ],")
        appendLine("  \"temperature\": $requestTemperature,")
        appendLine("  \"stream\": false")
        appendLine("}")
    }

    private fun readChatContent(json: String): String {
        val root = JsonParser(json).parseObject()
        val choices = root["choices"] as? List<*> ?: error("DeepSeek response does not contain choices.")
        val first = choices.firstOrNull() as? Map<*, *> ?: error("DeepSeek response contains no choices.")
        val message = first["message"] as? Map<*, *> ?: error("DeepSeek response does not contain message.")
        return message["content"] as? String ?: error("DeepSeek response does not contain message content.")
    }
}

object IndexReader {
    fun read(path: Path): IndexFile {
        val root = JsonParser(Files.readString(path, StandardCharsets.UTF_8)).parseObject()
        val chunks = root.list("chunks").map { value ->
            val chunk = value.asObject("chunk")
            val metadata = chunk.obj("metadata")
            IndexedChunk(
                metadata = ChunkMetadata(
                    source = metadata.string("source"),
                    title = metadata.string("title"),
                    section = metadata["section"] as? String,
                    chunkId = metadata.string("chunk_id")
                ),
                text = chunk.string("text"),
                embedding = chunk.list("embedding").map { number ->
                    (number as? Number)?.toDouble() ?: error("Embedding value must be a number.")
                }
            )
        }
        return IndexFile(
            document = root.string("document"),
            model = root.string("model"),
            strategy = root.string("strategy"),
            chunks = chunks
        )
    }
}

object JsonWriter {
    fun write(index: IndexFile): String = buildString {
        appendLine("{")
        appendLine("  \"document\": \"${escape(index.document)}\",")
        appendLine("  \"model\": \"${escape(index.model)}\",")
        appendLine("  \"strategy\": \"${escape(index.strategy)}\",")
        appendLine("  \"chunks\": [")
        index.chunks.forEachIndexed { chunkIndex, chunk ->
            appendLine("    {")
            appendLine("      \"metadata\": {")
            appendLine("        \"source\": \"${escape(chunk.metadata.source)}\",")
            appendLine("        \"title\": \"${escape(chunk.metadata.title)}\",")
            if (chunk.metadata.section == null) {
                appendLine("        \"section\": null,")
            } else {
                appendLine("        \"section\": \"${escape(chunk.metadata.section)}\",")
            }
            appendLine("        \"chunk_id\": \"${escape(chunk.metadata.chunkId)}\"")
            appendLine("      },")
            appendLine("      \"text\": \"${escape(chunk.text)}\",")
            append("      \"embedding\": [")
            append(chunk.embedding.joinToString(","))
            appendLine("]")
            append("    }")
            if (chunkIndex < index.chunks.lastIndex) appendLine(",") else appendLine()
        }
        appendLine("  ]")
        appendLine("}")
    }

    fun escape(value: String): String = buildString(value.length + 16) {
        value.forEach { ch ->
            when (ch) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> {
                    if (ch.code < 0x20) append("\\u%04x".format(ch.code)) else append(ch)
                }
            }
        }
    }

    fun readEmbedding(json: String): List<Double> {
        val key = """"embedding""""
        val keyIndex = json.indexOf(key)
        if (keyIndex < 0) error("Ollama response does not contain an embedding array.")
        val start = json.indexOf('[', keyIndex)
        val end = findMatchingBracket(json, start)
        if (start < 0 || end < 0) error("Malformed embedding array in Ollama response.")

        return json.substring(start + 1, end)
            .split(',')
            .mapNotNull { it.trim().takeIf(String::isNotEmpty)?.toDoubleOrNull() }
            .also {
                if (it.isEmpty()) error("Ollama returned an empty embedding.")
            }
    }

    private fun findMatchingBracket(json: String, start: Int): Int {
        if (start < 0) return -1
        var depth = 0
        for (i in start until json.length) {
            when (json[i]) {
                '[' -> depth++
                ']' -> {
                    depth--
                    if (depth == 0) return i
                }
            }
        }
        return -1
    }
}

private class JsonParser(private val json: String) {
    private var index = 0

    @Suppress("UNCHECKED_CAST")
    fun parseObject(): Map<String, Any?> {
        val value = parseValue()
        skipWhitespace()
        if (index != json.length) error("Unexpected trailing JSON content at position $index.")
        return value as? Map<String, Any?> ?: error("JSON root must be an object.")
    }

    private fun parseValue(): Any? {
        skipWhitespace()
        if (index >= json.length) error("Unexpected end of JSON.")
        return when (json[index]) {
            '{' -> parseObjectValue()
            '[' -> parseArray()
            '"' -> parseString()
            't' -> parseLiteral("true", true)
            'f' -> parseLiteral("false", false)
            'n' -> parseLiteral("null", null)
            '-', in '0'..'9' -> parseNumber()
            else -> error("Unexpected character '${json[index]}' at position $index.")
        }
    }

    private fun parseObjectValue(): Map<String, Any?> {
        expect('{')
        skipWhitespace()
        val result = linkedMapOf<String, Any?>()
        if (peek('}')) {
            index++
            return result
        }

        while (true) {
            val key = parseString()
            skipWhitespace()
            expect(':')
            result[key] = parseValue()
            skipWhitespace()
            when {
                peek(',') -> index++
                peek('}') -> {
                    index++
                    return result
                }
                else -> error("Expected ',' or '}' at position $index.")
            }
        }
    }

    private fun parseArray(): List<Any?> {
        expect('[')
        skipWhitespace()
        val result = mutableListOf<Any?>()
        if (peek(']')) {
            index++
            return result
        }

        while (true) {
            result += parseValue()
            skipWhitespace()
            when {
                peek(',') -> index++
                peek(']') -> {
                    index++
                    return result
                }
                else -> error("Expected ',' or ']' at position $index.")
            }
        }
    }

    private fun parseString(): String {
        expect('"')
        val result = StringBuilder()
        while (index < json.length) {
            val ch = json[index++]
            when (ch) {
                '"' -> return result.toString()
                '\\' -> {
                    if (index >= json.length) error("Unfinished escape sequence.")
                    result.append(
                        when (val escaped = json[index++]) {
                            '"' -> '"'
                            '\\' -> '\\'
                            '/' -> '/'
                            'b' -> '\b'
                            'f' -> '\u000C'
                            'n' -> '\n'
                            'r' -> '\r'
                            't' -> '\t'
                            'u' -> parseUnicodeEscape()
                            else -> error("Unsupported escape sequence '\\$escaped'.")
                        }
                    )
                }
                else -> result.append(ch)
            }
        }
        error("Unterminated JSON string.")
    }

    private fun parseUnicodeEscape(): Char {
        if (index + 4 > json.length) error("Invalid unicode escape.")
        val hex = json.substring(index, index + 4)
        index += 4
        return hex.toIntOrNull(16)?.toChar() ?: error("Invalid unicode escape: $hex")
    }

    private fun parseNumber(): Double {
        val start = index
        if (peek('-')) index++
        readDigits()
        if (peek('.')) {
            index++
            readDigits()
        }
        if (index < json.length && (json[index] == 'e' || json[index] == 'E')) {
            index++
            if (index < json.length && (json[index] == '+' || json[index] == '-')) index++
            readDigits()
        }
        return json.substring(start, index).toDoubleOrNull()
            ?: error("Invalid number at position $start.")
    }

    private fun readDigits() {
        val start = index
        while (index < json.length && json[index].isDigit()) index++
        if (start == index) error("Expected digit at position $index.")
    }

    private fun parseLiteral(literal: String, value: Any?): Any? {
        if (!json.startsWith(literal, index)) error("Expected '$literal' at position $index.")
        index += literal.length
        return value
    }

    private fun expect(ch: Char) {
        skipWhitespace()
        if (!peek(ch)) error("Expected '$ch' at position $index.")
        index++
    }

    private fun peek(ch: Char): Boolean = index < json.length && json[index] == ch

    private fun skipWhitespace() {
        while (index < json.length && json[index].isWhitespace()) index++
    }
}

private fun Map<String, Any?>.string(key: String): String =
    this[key] as? String ?: error("JSON field '$key' must be a string.")

private fun Map<String, Any?>.obj(key: String): Map<String, Any?> =
    this[key].asObject(key)

private fun Map<String, Any?>.list(key: String): List<Any?> =
    this[key] as? List<*> ?: error("JSON field '$key' must be an array.")

@Suppress("UNCHECKED_CAST")
private fun Any?.asObject(name: String): Map<String, Any?> =
    this as? Map<String, Any?> ?: error("JSON field '$name' must be an object.")
