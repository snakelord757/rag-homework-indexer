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
        val config = Cli.parse(args)
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

class CliException(message: String) : RuntimeException(message)

object Cli {
    fun parse(args: Array<String>): CliConfig {
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
        var model = "nomic-embed-text"
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

    fun printUsage() {
        println(
            """
            Usage:
              java -jar rag-indexer.jar <document-name> --strategy fixed [--size 1200] [--overlap 150] [--pages N]
              java -jar rag-indexer.jar <document-name> --strategy semantic [--max-chars 1600] [--threshold 0.72] [--pages N]

            Options:
              --model MODEL          Ollama embedding model, default: nomic-embed-text
              --ollama-url URL       Ollama base URL, default: http://localhost:11434

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
