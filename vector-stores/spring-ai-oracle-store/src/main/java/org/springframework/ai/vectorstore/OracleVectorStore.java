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

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
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
import org.springframework.beans.factory.InitializingBean;
import org.springframework.jdbc.core.BatchPreparedStatementSetter;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.PreparedStatementSetter;
import org.springframework.jdbc.core.RowMapper;

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

	public static final int OPENAI_EMBEDDING_DIMENSION = 1536;

	public static final int INVALID_EMBEDDING_DIMENSION = -1;

	public String VECTOR_TABLE = "vector_store";

	public int BATCH_SIZE = 100;

	private JdbcTemplate jdbcTemplate;

	EmbeddingModel embeddingClient;

	private int dimensions;

	private OracleDistanceType distanceType;

	private boolean removeExistingVectorStoreTable;

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

		COSINE("<=>", "COSINE_DISTANCE"), DOT("<#>", "INNER_PRODUCT"), EUCLIDEAN("<->", "L2_DISTANCE"),
		MANHATTAN(null, "L1_DISTANCE");

		public final String operator;

		public final String shorthandFunction;

		OracleDistanceType(String operator, String shorthandFunction) {
			this.operator = operator;
			this.shorthandFunction = shorthandFunction;
		}

	}

	public OracleVectorStore(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingClient) {
		this(jdbcTemplate, embeddingClient, INVALID_EMBEDDING_DIMENSION, OracleDistanceType.COSINE, false);
	}

	public OracleVectorStore(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingClient, int dimensions,
			OracleDistanceType distanceType, boolean removeExistingVectorStoreTable) {
		this.jdbcTemplate = jdbcTemplate;
		this.embeddingClient = embeddingClient;
		this.dimensions = dimensions;
		this.distanceType = distanceType;
		this.removeExistingVectorStoreTable = removeExistingVectorStoreTable;
	}

	@Override
	public void add(List<Document> documents) {

		int size = documents.size();

		this.jdbcTemplate.batchUpdate(
				"insert into " + this.VECTOR_TABLE + " (text, embeddings, metadata) values (?,?,?)",
				new BatchPreparedStatementSetter() {
					@Override
					public void setValues(PreparedStatement ps, int i) throws SQLException {

						var document = documents.get(i);
						var text = document.getContent();

						OracleJsonFactory factory = new OracleJsonFactory();
						OracleJsonObject jsonObj = factory.createObject();
						Map<String, Object> metaData = document.getMetadata();
						for (Map.Entry<String, Object> entry : metaData.entrySet()) {
							jsonObj.put(entry.getKey(), String.valueOf(entry.getValue()));
						}

						List<Double> vectorList = embeddingClient.embed(document);
						float[] embeddings = new float[vectorList.size()];
						for (int j = 0; j < vectorList.size(); j++) {
							embeddings[j] = vectorList.get(j).floatValue();
						}

						ps.setString(1, text);
						ps.setObject(2, embeddings, OracleType.VECTOR);
						ps.setObject(3, jsonObj, OracleType.JSON);

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

		List<Double> queryEmbeddings = embeddingClient.embed(request.getQuery());
		logger.debug("Embeddings size: " + queryEmbeddings.size());

		logger.debug("Distance metrics: " + this.distanceType);
		logger.debug("Distance metrics function: " + this.distanceType.name());

		int topK = request.getTopK();

		try {
			nearest = similaritySearchByMetrics(VECTOR_TABLE, queryEmbeddings, topK, this.distanceType.name());
		}
		catch (Exception e) {
			logger.error(e.toString());
		}

		List<Document> documents = new ArrayList<>();

		for (VectorData d : nearest) {
			OracleJsonObject metadata = d.getMetadata();
			Map<String, Object> map = new HashMap<>();
			for (String key : metadata.keySet()) {
				map.put(key, metadata.get(key).toString());
			}
			Document doc = new Document(d.getText(), map);
			documents.add(doc);

		}
		return documents;

	}

	List<VectorData> similaritySearchByMetrics(String vectortab, List<Double> vector, int topK,
			String distance_metrics_func) throws SQLException {
		List<VectorData> results = new ArrayList<>();
		float[] floatVector = new float[vector.size()];
		for (int i = 0; i < vector.size(); i++) {
			floatVector[i] = vector.get(i).floatValue();
		}

		try {

			String similaritySql = "select id, embeddings, metadata, text from " + vectortab + " order by "
					+ "vector_distance(embeddings, ?, " + distance_metrics_func + ")" + " fetch first ? rows only";

			results = jdbcTemplate.query(similaritySql, new PreparedStatementSetter() {
				public void setValues(java.sql.PreparedStatement ps) throws SQLException {
					ps.setObject(1, floatVector, OracleType.VECTOR);
					ps.setObject(2, topK, OracleType.NUMBER);
				}
			}, new RowMapper<VectorData>() {
				public VectorData mapRow(ResultSet rs, int rowNum) throws SQLException {
					return new VectorData(rs.getString("id"), rs.getObject("embeddings", float[].class),
							rs.getObject("text", String.class), rs.getObject("metadata", OracleJsonObject.class));
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
					        begin
					            execute immediate 'create table %s (
					            id number generated as identity,
					            text clob,
					            embeddings vector,
					            metadata json,
					            primary key (id))';
					        exception
					            when others then
					            if sqlcode != -942 then
					                raise;
					            end if;
					        end;
					""", this.VECTOR_TABLE));
			logger.debug("Create table " + this.VECTOR_TABLE);
		}
		catch (Exception e) {
			logger.error("Error creating table\n" + e.getMessage());
			throw (e);
		}

		return;
	}

}
