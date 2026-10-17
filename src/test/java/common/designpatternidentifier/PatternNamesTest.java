package common.designpatternidentifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Tests that every spelling of a pattern name maps onto the AbstractFactory convention. */
class PatternNamesTest {

    @ParameterizedTest
    @ValueSource(strings = {"AbstractFactory", "abstract_factory", "abstract-factory", "Abstract Factory", "abstractFactory", " abstract  factory "})
    void allSpellingsAgree(String spelling) {
        assertEquals("AbstractFactory", PatternNames.toKey(spelling));
        assertEquals("Abstract Factory", PatternNames.toDisplayName(spelling));
        assertEquals("abstract factory", PatternNames.toPlainText(spelling));
        assertEquals("abstractfactory", PatternNames.normalise(spelling));
    }

    @Test
    void singleWordPatterns() {
        assertEquals("Observer", PatternNames.toKey("observer"));
        assertEquals("Observer", PatternNames.toDisplayName("observer"));
        assertEquals("observer", PatternNames.toPlainText("Observer"));
    }

    @Test
    void plainTextMatchesTheOldSentenceForm() {
        // DesignPatterns.patternNameAsText used to be convertToPlainText("factory_method") = "factory method"
        assertEquals("factory method", PatternNames.toPlainText(PatternNames.FACTORY_METHOD));
    }

    @Test
    void constantsAreAlreadyCanonicalKeys() {
        for (String key : new String[] {PatternNames.SINGLETON, PatternNames.FACTORY_METHOD, PatternNames.ABSTRACT_FACTORY,
                PatternNames.ADAPTER, PatternNames.DECORATOR, PatternNames.FACADE, PatternNames.OBSERVER,
                PatternNames.VISITOR, PatternNames.MEMENTO}) {
            assertEquals(key, PatternNames.toKey(key));
        }
    }

    @Test
    void sameIgnoresSpelling() {
        assertTrue(PatternNames.same("factory_method", "FactoryMethod"));
        assertTrue(PatternNames.same("Factory Method", "factory-method"));
    }
}
