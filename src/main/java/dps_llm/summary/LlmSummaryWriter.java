package dps_llm.summary;

import java.io.BufferedWriter;
// import java.io.File;                      // CHANGE: directory creation moved to CsvUtils.openUtf8Writer
import java.io.IOException;
// import java.nio.charset.StandardCharsets; // CHANGE: see CsvUtils.openUtf8Writer
// import java.nio.file.Files;               // CHANGE: see CsvUtils.openUtf8Writer
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import common.utils.CsvUtils;
import common.utils.ProjectPathFormatter;

/**
 * CSV writer for LLM-generated summaries.
 * <p>
 * This class manages the output CSV file that stores generated summaries alongside
 * project and file metadata. It handles CSV formatting, escaping, and ensures
 * thread-safe write operations.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Create and initialize the CSV output file with headers</li>
 *   <li>Write summary rows with proper CSV escaping</li>
 *   <li>Format project paths using ProjectPathFormatter</li>
 *   <li>Write summaries in full (CHANGE: the old 600-character truncation was removed)</li>
 *   <li>Ensure thread-safe write operations</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class LlmSummaryWriter implements AutoCloseable {

    private final BufferedWriter writer;
    // CHANGE: keys (project, folder, file) of the rows already in the CSV when resuming; empty otherwise.
    private final Set<String> existingKeys = new HashSet<>();

    /**
     * Constructs a new CSV writer and initializes the output file.
     * <p>
     * Creates the output directory if it doesn't exist and writes the CSV header row.
     * </p>
     * 
     * @param outputPath the path to the output CSV file
     * @throws IOException if directory creation or file writing fails
     * @throws IllegalArgumentException if outputPath is null or blank
     */
    public LlmSummaryWriter(String outputPath) throws IOException {
        if (outputPath == null || outputPath.isBlank()) {
            throw new IllegalArgumentException("outputPath must not be null or blank");
        }
        // CHANGE: directory creation and the UTF-8 writer are shared with the NLG/SWUM writers via CsvUtils.
        // File file = new File(outputPath);
        // File parent = file.getParentFile();
        // if (parent != null && !parent.exists()) {
        //     if (!parent.mkdirs() && !parent.exists()) {
        //         throw new IOException("Failed to create output directory: " + parent.getAbsolutePath());
        //     }
        // }
        // this.writer = Files.newBufferedWriter(file.toPath(), StandardCharsets.UTF_8);
        // writer.write("Project,Folder Name,File Name,Summary\n");
        this.writer = CsvUtils.openUtf8Writer(outputPath);
        writer.write(CsvUtils.SUMMARY_HEADER + "\n");
    }

    /**
     * Constructs a writer that either starts a fresh CSV or, with {@code resume}, keeps the rows
     * already in it and appends new ones.
     * <p>
     * CHANGE: added for {@code --resume}. A normal run truncates the CSV, so a run that stopped
     * part-way (e.g. 429 rate limits) had to request every class again. When resuming, the existing
     * rows are read first and {@link #hasRow} lets the caller skip those classes without an API call.
     * </p>
     *
     * @param outputPath the path to the output CSV file
     * @param resume true to keep existing rows and append; false to behave like {@link #LlmSummaryWriter(String)}
     * @throws IOException if reading or opening the file fails
     */
    public LlmSummaryWriter(String outputPath, boolean resume) throws IOException {
        if (outputPath == null || outputPath.isBlank()) {
            throw new IllegalArgumentException("outputPath must not be null or blank");
        }
        Path path = Path.of(outputPath);
        if (!resume || !Files.isRegularFile(path) || Files.size(path) == 0) {
            this.writer = CsvUtils.openUtf8Writer(outputPath);
            writer.write(CsvUtils.SUMMARY_HEADER + "\n");
            return;
        }
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        for (int i = 1; i < lines.size(); i++) {
            List<String> fields = parseRow(lines.get(i));
            if (fields.size() >= 3) {
                existingKeys.add(key(fields.get(0), fields.get(1), fields.get(2)));
            }
        }
        boolean endsWithNewline;
        try (RandomAccessFile file = new RandomAccessFile(path.toFile(), "r")) {
            file.seek(file.length() - 1);
            endsWithNewline = file.read() == '\n';
        }
        this.writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8, StandardOpenOption.APPEND);
        if (!endsWithNewline) {
            writer.write("\n");
        }
    }

    /** Number of rows found in the CSV when resuming (0 for a fresh CSV). */
    public int existingRowCount() {
        return existingKeys.size();
    }

    /**
     * Returns true if the CSV already held a row for this file when the writer was opened in resume mode.
     *
     * @param projectIdentifier the project identifier (formatted as "ProjectName/FolderName")
     * @param filename the source file name
     */
    public boolean hasRow(String projectIdentifier, String filename) {
        if (existingKeys.isEmpty() || projectIdentifier == null || filename == null) {
            return false;
        }
        ProjectPathFormatter.Parts parts = ProjectPathFormatter.split(projectIdentifier);
        return existingKeys.contains(key(parts.projectName(), parts.folderName(), filename));
    }

    private static String key(String project, String folder, String file) {
        return project.trim() + "\u0000" + folder.trim() + "\u0000" + file.trim();
    }

    /** Splits one row written by {@link CsvUtils#formatRow} (quoted fields, doubled quotes) into fields. */
    static List<String> parseRow(String line) {
        List<String> fields = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean quoted = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (quoted) {
                if (c == '"' && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else if (c == '"') {
                    quoted = false;
                } else {
                    current.append(c);
                }
            } else if (c == '"') {
                quoted = true;
            } else if (c == ',') {
                fields.add(current.toString());
                current.setLength(0);
            } else {
                current.append(c);
            }
        }
        fields.add(current.toString());
        return fields;
    }

    /**
     * Writes a single summary row to the CSV file.
     * <p>
     * The method is synchronized to ensure thread-safe writes. Summaries are written in full.
     * </p>
     * 
     * @param projectIdentifier the project identifier (formatted as "ProjectName/FolderName")
     * @param filename the source file name
     * @param summary the generated summary text
     * @throws IOException if writing fails
     * @throws IllegalArgumentException if projectIdentifier or filename is null
     */
    public synchronized void writeRow(String projectIdentifier, String filename, String summary) throws IOException {
        if (projectIdentifier == null) {
            throw new IllegalArgumentException("projectIdentifier must not be null");
        }
        if (filename == null) {
            throw new IllegalArgumentException("filename must not be null");
        }
        // CHANGE: rows are built by CsvUtils so all three pipelines escape identically. Two behaviour
        // fixes: summaries are no longer truncated at 600 characters, and rows end with "\n" instead of
        // %n (which wrote "\r\n" on Windows while the NLG and SWUM CSVs used "\n").
        // Original implementation:
        // String safeSummary = clean(summary);
        // ProjectPathFormatter.Parts parts = ProjectPathFormatter.split(projectIdentifier);
        // writer.write(String.format("\"%s\",\"%s\",\"%s\",\"%s\"%n",
        //         escape(parts.projectName()),
        //         escape(parts.folderName()),
        //         escape(filename),
        //         safeSummary));
        ProjectPathFormatter.Parts parts = ProjectPathFormatter.split(projectIdentifier);
        writer.write(CsvUtils.formatRow(parts.projectName(), parts.folderName(), filename, summary));
        writer.flush();
    }

    // CHANGE: replaced by CsvUtils.escapeField (which no longer truncates).
    // private String clean(String text) {
    //     if (text == null) {
    //         return "";
    //     }
    //     String cleaned = text.replace("\"", "\"\"")
    //             .replace("\r", " ")
    //             .replace("\n", " ")
    //             .trim();
    //     if (cleaned.length() > 600) {
    //         cleaned = cleaned.substring(0, 600) + "...";
    //     }
    //     return cleaned;
    // }
    //
    // private String escape(String value) {
    //     if (value == null) {
    //         return "";
    //     }
    //     return value.replace("\"", "\"\"");
    // }

    /**
     * Closes the CSV writer and flushes any remaining data.
     * 
     * @throws IOException if closing the writer fails
     */
    @Override
    public void close() throws IOException {
        writer.close();
    }
}
