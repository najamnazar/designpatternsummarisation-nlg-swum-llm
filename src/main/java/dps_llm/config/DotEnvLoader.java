package dps_llm.config;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Minimalist .env file parser for loading configuration from environment files.
 * <p>
 * This utility class reads key-value pairs from .env files to supply API credentials
 * and configuration without committing sensitive data to version control. It supports
 * basic .env syntax including comments and blank lines.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Parse .env files in KEY=VALUE format</li>
 *   <li>Skip comments (lines starting with #) and blank lines</li>
 *   <li>Return an immutable map of configuration values</li>
 *   <li>Handle missing files gracefully by returning an empty map</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public final class DotEnvLoader {

    private DotEnvLoader() {
    }

    /**
     * Loads configuration key-value pairs from a .env file.
     * <p>
     * Lines starting with # are treated as comments. Blank lines are ignored.
     * Each configuration line must be in the format KEY=VALUE.
     * </p>
     * 
     * @param path the path to the .env file
     * @return a map of configuration keys to values, or empty map if file doesn't exist
     * @throws IOException if file reading fails
     * @throws NullPointerException if path is null
     */
    public static Map<String, String> load(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        if (!Files.exists(path)) {
            return Collections.emptyMap();
        }
        Map<String, String> values = new HashMap<>();
        for (String rawLine : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            String line = rawLine.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            int equalsIndex = line.indexOf('=');
            if (equalsIndex <= 0) {
                continue;
            }
            String key = line.substring(0, equalsIndex).trim();
            String value = line.substring(equalsIndex + 1).trim();
            // CHANGE: strip one pair of matching surrounding quotes, as python-dotenv does. Without this,
            // OPENROUTER_API_KEY="sk-..." worked for the Python scripts but the Java pipeline sent the
            // quotes as part of the key. Also accept an optional "export " prefix (shell-style .env files).
            // values.put(key, value);
            if (key.startsWith("export ")) {
                key = key.substring("export ".length()).trim();
            }
            values.put(key, stripMatchingQuotes(value));
        }
        return values;
    }

    /**
     * Removes one pair of matching single or double quotes surrounding a value.
     * CHANGE: added so quoted .env values behave the same in Java and Python.
     *
     * @param value the raw value (already trimmed)
     * @return the value without surrounding quotes
     */
    static String stripMatchingQuotes(String value) {
        if (value.length() >= 2) {
            char first = value.charAt(0);
            char last = value.charAt(value.length() - 1);
            if ((first == '"' || first == '\'') && first == last) {
                return value.substring(1, value.length() - 1);
            }
        }
        return value;
    }
}
