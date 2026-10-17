package common.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.StringJoiner;

/**
 * Shared CSV helpers for the NLG, SWUM and LLM summary writers.
 * <p>
 * CHANGE (refactor + fixes): {@code Summarise}, {@code SWUMSummarizer} and
 * {@code LlmSummaryWriter} each had their own copy of this logic, and the copies disagreed:
 * <ul>
 *   <li>The summary was cut at 1000 (NLG, SWUM) or 600 (LLM) characters and given a
 *       {@code ...} suffix. 21 of the 150 NLG summaries were truncated this way, which
 *       changed the length and similarity figures. Summaries are no longer truncated.</li>
 *   <li>Truncating after escaping could split a doubled quote ({@code ""}) and corrupt the
 *       row. Escaping now runs last, on the full text.</li>
 *   <li>NLG and SWUM used {@code new FileWriter(file)}, i.e. the platform default charset
 *       (not guaranteed UTF-8 on Windows before JDK 18). All writers now use UTF-8.</li>
 *   <li>The LLM writer ended rows with {@code %n} ({@code \r\n} on Windows) while the others
 *       used {@code \n}. All rows now end with {@code \n}.</li>
 * </ul>
 * </p>
 */
public final class CsvUtils {

    /** Header shared by every summary CSV. */
    public static final String SUMMARY_HEADER = "Project,Folder Name,File Name,Summary";

    private CsvUtils() {
        // Utility class - prevent instantiation
    }

    /**
     * Opens a UTF-8 writer for {@code path}, creating parent directories and truncating any
     * existing file.
     *
     * @param path output file path
     * @return a buffered UTF-8 writer
     * @throws IOException if the directory or file cannot be created
     */
    public static BufferedWriter openUtf8Writer(String path) throws IOException {
        File file = new File(path);
        File parent = file.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs() && !parent.exists()) {
            throw new IOException("Failed to create output directory: " + parent.getAbsolutePath());
        }
        return Files.newBufferedWriter(file.toPath(), StandardCharsets.UTF_8);
    }

    /**
     * Prepares one value for a quoted CSV field: newlines become spaces, the text is trimmed,
     * and embedded quotes are doubled. The text is never truncated.
     *
     * @param value raw value, may be null
     * @return escaped value without the surrounding quotes
     */
    public static String escapeField(String value) {
        if (value == null) {
            return "";
        }
        String flattened = value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").trim();
        return flattened.replace("\"", "\"\"");
    }

    /**
     * Builds one CSV row in which every field is quoted and escaped, ending with {@code \n}.
     *
     * @param fields raw field values
     * @return the formatted row
     */
    public static String formatRow(String... fields) {
        StringJoiner joiner = new StringJoiner(",", "", "\n");
        for (String field : fields) {
            joiner.add("\"" + escapeField(field) + "\"");
        }
        return joiner.toString();
    }
}
