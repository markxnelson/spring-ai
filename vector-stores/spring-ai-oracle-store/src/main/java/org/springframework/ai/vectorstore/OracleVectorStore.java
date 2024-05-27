/*
 * Copyright 2023 - 2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springframework.ai.vectorstore;

import java.math.BigDecimal;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.SQLSyntaxErrorException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.filter.FilterExpressionConverter;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.dao.DataAccessException;
import org.springframework.jdbc.core.BatchPreparedStatementSetter;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.PreparedStatementSetter;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.util.StringUtils;

//Oracle DB
import oracle.jdbc.OracleType;
import oracle.sql.json.OracleJsonFactory;
import oracle.sql.json.OracleJsonObject;

/**
 * Uses Oracle Database 23ai Vector feature to store Spring AI vectors.
 *
 * @author Corrado De Bari
 * @author Mark Nelson
 */
public class OracleVectorStore implements VectorStore, InitializingBean {

	private static final Logger logger = LoggerFactory.getLogger(OracleVectorStore.class);

	public static final byte DEFAULT_ACCURACY = 90;

	public static final OracleIndexType DEFAULT_INDEX_TYPE = OracleIndexType.NONE;

	public static final int OPENAI_EMBEDDING_DIMENSION_SIZE = 1536;

	public static final int INVALID_EMBEDDING_DIMENSION = -1;

	public String VECTOR_TABLE = "vector_store";

	public final FilterExpressionConverter filterExpressionConverter = new OracleFilterExpressionConverter();

	public int BATCH_SIZE = 100;

	private JdbcTemplate jdbcTemplate;

	EmbeddingModel embeddingModel;

	private int dimensions;

	private OracleDistanceType distanceType;

	private boolean removeExistingVectorStoreTable;

	private OracleIndexType indexType;

	private byte accuracy;

	public enum OracleIndexType {

		/**
		 * Performs exact search.
		 */
		NONE,
		/**
		 * Inverted File Flat (IVF) is a form of Neighbor Partition Vector index. It is a
		 * partition-based index that achieves search efficiency by narrowing the search
		 * area through the use of neighbor partitions or clusters.
		 */
		IVF,
		/**
		 * A Hierarchical Navigable Small World Graph (HNSW) is a form of In-Memory
		 * Neighbor Graph vector index. It is a very efficient index for vector
		 * approximate similarity search.
		 */
		HNSW;

	}

	public enum OracleDistanceType {

		/**
		 * One of the most widely used similarity metric, especially in natural language
		 * processing (NLP), is cosine similarity, which measures the cosine of the angle
		 * between two vectors.
		 */
		COSINE,
		/**
		 * The dot product similarity of two vectors can be viewed as multiplying the size
		 * of each vector by the cosine of their angle. The corresponding geometrical
		 * interpretation of this definition is equivalent to multiplying the size of one
		 * of the vectors by the size of the projection of the second vector onto the
		 * first one, or vice versa.
		 */
		DOT,
		/**
		 * Euclidean distance reflects the distance between each of the vectors'
		 * coordinates being compared—basically the straight-line distance between two
		 * vectors. This is calculated using the Pythagorean theorem applied to the
		 * vector's coordinates.
		 */
		EUCLIDEAN

	}

	/**
	 * Creates an OracleVectorStore
	 * @param jdbcTemplate JdbcTemplate bean used to access the underlying database
	 * @param embeddingClient EmbeddingModel bean used to convert the documents into
	 * vectors
	 */
	public OracleVectorStore(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingClient) {
		this(jdbcTemplate, embeddingClient, INVALID_EMBEDDING_DIMENSION, OracleDistanceType.COSINE, false,
				DEFAULT_INDEX_TYPE, DEFAULT_ACCURACY);
	}

	/**
	 * Creates an OracleVectorStore
	 * @param jdbcTemplate JdbcTemplate bean used to access the underlying database
	 * @param embeddingClient EmbeddingModel bean used to convert the documents into
	 * vectors
	 * @param dimensions FIXME : not used
	 */
	public OracleVectorStore(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingClient, int dimensions) {
		this(jdbcTemplate, embeddingClient, dimensions, OracleDistanceType.COSINE, false, DEFAULT_INDEX_TYPE,
				DEFAULT_ACCURACY);
	}

	/**
	 * Creates an OracleVectorStore
	 * @param jdbcTemplate JdbcTemplate bean used to access the underlying database
	 * @param embeddingClient EmbeddingModel bean used to convert the documents into
	 * vectors
	 * @param dimensions FIXME : not used
	 * @param distanceType Distance function that will be used when searching, and, if an
	 * index is created, it will use this distance type.
	 * @param removeExistingVectorStoreTable Indicates whether the existing "vector_store"
	 * table should be deleted.
	 * @param indexType The index type, if set to a value different to NONE, an index will
	 * be created using the distanceType and accuracy provided.
	 * @param accuracy Percentage value (between 0 and 100) representing the accuracy of
	 * the index.
	 */
	public OracleVectorStore(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingClient, int dimensions,
			OracleDistanceType distanceType, boolean removeExistingVectorStoreTable, OracleIndexType indexType,
			byte accuracy) {
		this.jdbcTemplate = jdbcTemplate;
		this.embeddingModel = embeddingClient;
		this.dimensions = dimensions;
		this.distanceType = distanceType;
		this.removeExistingVectorStoreTable = removeExistingVectorStoreTable;
		this.indexType = indexType;
		if (accuracy < 0 || accuracy > 100) {
			logger.warn("Invalid accuracy value provided, falling back to default value.");
			this.accuracy = DEFAULT_ACCURACY;
		}
		else {
			this.accuracy = accuracy;
		}
	}

	@Override
	public void add(List<Document> documents) {

		int size = documents.size();

		this.jdbcTemplate.batchUpdate(String.format("""
				merge into %s
				using dual
				on (id = ?)
				when matched then update set text = ?, embeddings = ?, metadata = ?
				when not matched then insert (id, text, embeddings, metadata) values (?,?,?,?)
				""", this.VECTOR_TABLE), new BatchPreparedStatementSetter() {
			@Override
			public void setValues(PreparedStatement ps, int i) throws SQLException {

				var document = documents.get(i);
				var id = document.getId();
				var text = document.getContent();

				OracleJsonFactory factory = new OracleJsonFactory();
				OracleJsonObject jsonObj = factory.createObject();
				Map<String, Object> metaData = document.getMetadata();
				for (Map.Entry<String, Object> entry : metaData.entrySet()) {
					jsonObj.put(entry.getKey(), String.valueOf(entry.getValue()));
				}

				List<Double> vectorList = embeddingModel.embed(document);
				float[] embeddings = new float[vectorList.size()];
				for (int j = 0; j < vectorList.size(); j++) {
					embeddings[j] = vectorList.get(j).floatValue();
				}

				ps.setString(1, id);
				ps.setString(2, text);
				ps.setObject(3, embeddings, OracleType.VECTOR);
				ps.setObject(4, jsonObj, OracleType.JSON);
				ps.setString(5, id);
				ps.setString(6, text);
				ps.setObject(7, embeddings, OracleType.VECTOR);
				ps.setObject(8, jsonObj, OracleType.JSON);

			}

			@Override
			public int getBatchSize() {
				return size;
			}
		});

	}

	@Override
	public Optional<Boolean> delete(List<String> idList) {

		String sql = "delete from " + this.VECTOR_TABLE + " where id = ?";
		int count[][] = jdbcTemplate.batchUpdate(sql, idList, BATCH_SIZE, (ps, argument) -> {
			ps.setString(1, argument);
		});

		int sum = Arrays.stream(count).flatMapToInt(Arrays::stream).sum();
		logger.debug("Deleted " + sum + " records");

		return Optional.of(sum == idList.size());
	}

	@Override
	public List<Document> similaritySearch(SearchRequest request) {

		List<VectorData> nearest = new ArrayList<>();

		logger.debug("Requested query " + request.getQuery());

		List<Double> queryEmbeddings = embeddingModel.embed(request.getQuery());
		logger.debug("Embeddings size: " + queryEmbeddings.size());

		logger.debug("Distance metrics: " + this.distanceType);
		logger.debug("Distance metrics function: " + this.distanceType.name());

		int topK = request.getTopK();

		String nativeFilterExpression = (request.getFilterExpression() != null)
				? this.filterExpressionConverter.convertExpression(request.getFilterExpression()) : "";

		String jsonPathFilter = "";

		if (StringUtils.hasText(nativeFilterExpression)) {
			jsonPathFilter = " where json_exists(metadata, '$?(" + nativeFilterExpression + ")') ";
		}

		double distance = 1 - request.getSimilarityThreshold();

		try {
			nearest = similaritySearchByMetrics(VECTOR_TABLE, queryEmbeddings, distance, topK, this.distanceType.name(),
					jsonPathFilter);
		}
		catch (Exception e) {
			logger.error(e.toString());
		}

		List<Document> documents = new ArrayList<>();

		for (VectorData d : nearest) {
			OracleJsonObject metadata = d.getMetadata();
			Map<String, Object> map = new HashMap<>();
			for (String key : metadata.keySet()) {
				map.put(key, metadata.get(key).toString().replaceAll("\"", ""));
			}
			// add distance to the metadata map
			map.put("distance", d.getDistance());
			System.out.println("Metadata map: " + map);
			Document doc = new Document(d.getId(), d.getText(), map);
			documents.add(doc);

		}
		return documents;

	}

	List<VectorData> similaritySearchByMetrics(String vectortab, List<Double> vector, double distance, int topK,
			String distance_metrics_func, String jsonPathFilter) throws SQLException {
		List<VectorData> results = new ArrayList<>();
		float[] floatVector = new float[vector.size()];
		for (int i = 0; i < vector.size(); i++) {
			floatVector[i] = vector.get(i).floatValue();
		}

		System.out.println("DISTANCE = " + distance);

		try {

			String similaritySql = String.format("""
							select * from (
								select id, embeddings, metadata, text,
								vector_distance(embeddings, ?, %s) distance
								from %s
								%s
								order by distance
							)
							where distance <= ?
							fetch first ? rows only
					""", distance_metrics_func, vectortab, jsonPathFilter);

			results = jdbcTemplate.query(similaritySql, new PreparedStatementSetter() {
				public void setValues(java.sql.PreparedStatement ps) throws SQLException {
					ps.setObject(1, floatVector, OracleType.VECTOR);
					ps.setObject(2, distance, OracleType.NUMBER);
					ps.setObject(3, topK, OracleType.NUMBER);
				}
			}, new RowMapper<VectorData>() {
				public VectorData mapRow(ResultSet rs, int rowNum) throws SQLException {
					return new VectorData(rs.getString("id"), rs.getObject("embeddings", double[].class),
							rs.getObject("text", String.class), rs.getObject("metadata", OracleJsonObject.class),
							((BigDecimal) rs.getObject("distance", BigDecimal.class)).doubleValue());
				}
			});

		}
		catch (Exception e) {
			logger.error(e.getMessage());
		}
		return results;
	}

	// ---------------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------------
	@Override
	public void afterPropertiesSet() throws Exception {
		if (removeExistingVectorStoreTable) {
			try {
				logger.debug("Dropping table " + this.VECTOR_TABLE + " because removeExistingVectorStoreTable = true.");

				jdbcTemplate.execute(String.format("""
						        begin
						          execute immediate 'drop table %s cascade constraints';
						        exception
						          when others then
						            if sqlcode != -942 then
						              raise;
						            end if;
						        end;
						""", this.VECTOR_TABLE));
			}
			catch (Exception e) {
				logger.error("Error dropping table " + this.VECTOR_TABLE + " \n" + e.getMessage());
				throw (e);
			}
		}

		try {
			this.jdbcTemplate.execute(String.format("""
					        create table %s (
					            id varchar2(36),
					            text clob,
					            embeddings vector,
					            metadata json,
					            primary key (id))
					""", this.VECTOR_TABLE));
			logger.debug("Create table " + this.VECTOR_TABLE);
		}
		catch (DataAccessException e) {
			if ((e.getCause() instanceof SQLSyntaxErrorException)
					&& (((SQLSyntaxErrorException) e.getCause()).getErrorCode() == 955)) {
				logger.info("Using existing table " + this.VECTOR_TABLE);
			}
			else {
				logger.error("Error creating table\n" + e.getMessage());
				throw (e);
			}
		}

		if (indexType != OracleIndexType.NONE) {
			String indexTypeSQL = switch (indexType) {
				case HNSW -> "ORGANIZATION INMEMORY NEIGHBOR GRAPH";
				case IVF -> "ORGANIZATION NEIGHBOR PARTITIONS";
				default -> "";
			};
			try {
				String createVectorIndexStatement = String.format("""
						CREATE VECTOR INDEX %s_%s_idx ON %s ( embeddings )
									%s
									WITH DISTANCE %s
									WITH TARGET ACCURACY %d
									""", indexType.toString().toLowerCase(), VECTOR_TABLE.toLowerCase(), VECTOR_TABLE,
						indexTypeSQL, distanceType, accuracy);
				this.jdbcTemplate.execute(createVectorIndexStatement);
				logger.debug(String.format("Create index  %s_%s_idx ", indexType.toString().toLowerCase(),
						VECTOR_TABLE.toLowerCase()));
			}
			catch (DataAccessException e) {
				logger.error("Error creating index\n" + e.getMessage());
				throw (e);
			}
		}

		return;
	}

	int embeddingDimensions() {
		// The manually set dimensions have precedence over the computed one.
		if (this.dimensions > 0) {
			return this.dimensions;
		}

		try {
			int embeddingDimensions = this.embeddingModel.dimensions();
			if (embeddingDimensions > 0) {
				return embeddingDimensions;
			}
		}
		catch (Exception e) {
			logger.warn("Failed to obtain the embedding dimensions from the embedding model and fall backs to default:"
					+ OPENAI_EMBEDDING_DIMENSION_SIZE);
		}
		return OPENAI_EMBEDDING_DIMENSION_SIZE;
	}

}
