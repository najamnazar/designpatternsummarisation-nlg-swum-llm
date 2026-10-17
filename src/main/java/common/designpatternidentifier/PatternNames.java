package common.designpatternidentifier;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Single source of truth for how design-pattern names are written.
 * <p>
 * CHANGE (naming consistency): pattern names used to appear in several spellings: input
 * folders ({@code Abstract Factory}, {@code abstract-factory}, {@code AbstractFactory}), Java
 * keys ({@code abstract_factory}) and summary text ({@code abstract_factory: ...}). The project
 * now uses one convention, matching the "Design Pattern" column of
 * {@code input/ground-truth/DPS_Human_Summaries.csv}:
 * <ul>
 *   <li><b>Key</b> (folders, JSON keys, CSV labels): PascalCase without spaces or hyphens,
 *       e.g. {@code AbstractFactory}. See {@link #toKey(String)}.</li>
 *   <li><b>Display name</b> (pattern names written into summary prose): words separated by
 *       spaces, e.g. {@code Abstract Factory}, as in the human and LLM summaries.
 *       See {@link #toDisplayName(String)}.</li>
 *   <li><b>Plain text</b> (lower-case noun used inside NLG sentences): e.g.
 *       {@code abstract factory}, unchanged from before. See {@link #toPlainText(String)}.</li>
 * </ul>
 * Every method accepts any of the spellings above, so older JSON output with snake_case keys
 * is still understood.
 * </p>
 */
public final class PatternNames {

    public static final String SINGLETON = "Singleton";
    public static final String FACTORY_METHOD = "FactoryMethod";
    public static final String ABSTRACT_FACTORY = "AbstractFactory";
    public static final String ADAPTER = "Adapter";
    public static final String DECORATOR = "Decorator";
    public static final String FACADE = "Facade";
    public static final String OBSERVER = "Observer";
    public static final String VISITOR = "Visitor";
    public static final String MEMENTO = "Memento";

    private PatternNames() {
        // Utility class - prevent instantiation
    }

    /**
     * Splits a pattern name in any supported spelling into lower-case words.
     * {@code "AbstractFactory"}, {@code "abstract_factory"}, {@code "abstract-factory"} and
     * {@code "Abstract Factory"} all give {@code [abstract, factory]}.
     */
    private static List<String> words(String name) {
        List<String> result = new ArrayList<>();
        if (name == null) {
            return result;
        }
        String spaced = name.trim()
                .replaceAll("(?<=\\p{Ll})(?=\\p{Lu})", " ") // camelCase boundary
                .replaceAll("[^\\p{L}\\p{N}]+", " ");        // _ - / and spaces
        for (String word : spaced.trim().split("\\s+")) {
            if (!word.isEmpty()) {
                result.add(word.toLowerCase(Locale.ROOT));
            }
        }
        return result;
    }

    private static String capitalise(String word) {
        return word.isEmpty() ? word : Character.toUpperCase(word.charAt(0)) + word.substring(1);
    }

    /**
     * Canonical key form, e.g. {@code AbstractFactory}.
     *
     * @param name a pattern name in any supported spelling
     * @return PascalCase name without separators
     */
    public static String toKey(String name) {
        StringBuilder key = new StringBuilder();
        for (String word : words(name)) {
            key.append(capitalise(word));
        }
        return key.toString();
    }

    /**
     * Readable name for summary prose, e.g. {@code Abstract Factory}.
     *
     * @param name a pattern name in any supported spelling
     * @return capitalised words separated by single spaces
     */
    public static String toDisplayName(String name) {
        List<String> capitalised = new ArrayList<>();
        for (String word : words(name)) {
            capitalised.add(capitalise(word));
        }
        return String.join(" ", capitalised);
    }

    /**
     * Lower-case noun used inside NLG sentences, e.g. {@code abstract factory}. This reproduces
     * what {@code Utils.convertToPlainText("abstract_factory")} produced before the keys changed.
     *
     * @param name a pattern name in any supported spelling
     * @return lower-case words separated by single spaces
     */
    public static String toPlainText(String name) {
        return String.join(" ", words(name));
    }

    /**
     * Spelling-independent comparison form, e.g. {@code abstractfactory}. Use it for equality
     * checks so that old and new spellings compare equal.
     *
     * @param name a pattern name in any supported spelling
     * @return lower-case letters and digits only
     */
    public static String normalise(String name) {
        return String.join("", words(name));
    }

    /**
     * True when two names refer to the same pattern regardless of spelling.
     */
    public static boolean same(String a, String b) {
        return normalise(a).equals(normalise(b));
    }
}
