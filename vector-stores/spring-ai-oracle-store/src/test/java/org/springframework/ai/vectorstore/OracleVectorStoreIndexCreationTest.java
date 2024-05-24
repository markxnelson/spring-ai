package org.springframework.ai.vectorstore;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.boot.test.context.runner.ContextConsumer;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.PreparedStatementSetter;
import org.springframework.jdbc.core.RowMapper;

import javax.sql.DataSource;
import java.util.List;

public class OracleVectorStoreIndexCreationTest {

  @ParameterizedTest
  @CsvSource({"COSINE,HNSW,70", "DOT,HNSW,70", "EUCLIDEAN,HNSW,70",
      "COSINE,IVF,70", "DOT,IVF,70", "EUCLIDEAN,IVF,70",
      "COSINE,HNSW,110", "DOT,HNSW,110", "EUCLIDEAN,HNSW,110",
      "COSINE,IVF,-1", "DOT,IVF,-1", "EUCLIDEAN,IVF,-1"})
  public void testDotConfig(String sDistanceType, String sIndexType, byte accuracy) throws Exception {
    JdbcTemplate jdbcTemplate = Mockito.mock(JdbcTemplate.class);
    EmbeddingModel embeddingModel = Mockito.mock(EmbeddingModel.class);

    OracleVectorStore.OracleDistanceType distanceType = OracleVectorStore.OracleDistanceType.valueOf(sDistanceType);
    OracleVectorStore.OracleIndexType indexType = OracleVectorStore.OracleIndexType.valueOf(sIndexType);
    String indexTypeSQL = switch (indexType) {
      case HNSW -> "ORGANIZATION INMEMORY NEIGHBOR GRAPH";
      case IVF -> "ORGANIZATION NEIGHBOR PARTITIONS";
      default -> "";
    };

    OracleVectorStore vectorStore = new OracleVectorStore(jdbcTemplate, embeddingModel,
        OracleVectorStore.INVALID_EMBEDDING_DIMENSION, distanceType,
        false, indexType, accuracy);
    vectorStore.afterPropertiesSet();
    Mockito.verify(jdbcTemplate).execute(Mockito.contains(indexTypeSQL));
    Mockito.verify(jdbcTemplate).execute(Mockito.contains(distanceType.toString()));
    if (accuracy < 0 || accuracy > 100) {
      Mockito.verify(jdbcTemplate).execute(Mockito.contains(String.valueOf(OracleVectorStore.DEFAULT_ACCURACY)));
    } else {
      Mockito.verify(jdbcTemplate).execute(Mockito.contains(String.valueOf(accuracy)));
    }
  }

}
