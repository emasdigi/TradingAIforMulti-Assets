from confluent_kafka import (
    Consumer,
    Producer,
    KafkaError,
    KafkaException,
    TopicPartition,
    OFFSET_END,
)
from config.config import logger
import time


def create_consumer(topic: list, CONFIG: dict, offset="earliest") -> object:
    """initiate KAFKA consumer

    Args:
        topic (string): KAFKA topic
        CONFIG (dict): KAFKA config info

    Returns:
        consumer object: KAFKA consumer to receive messages
    """
    consumer = Consumer(
        {
            "bootstrap.servers": CONFIG["bootstrapServers"],
            "group.id": CONFIG["consumer"]["groupId"],
            "enable.auto.commit": False,
            "auto.offset.reset": offset,  #'latest'
            "security.protocol": "SSL",
            "ssl.ca.location": "ca.pem",
            "ssl.certificate.location": "service.cert",
            "ssl.key.location": "service.key",
        }
    )
    consumer.subscribe(topic)
    return consumer


def create_producer(CONFIG) -> object:
    """initiate KAFKA producer

    Args:
        CONFIG (dict): KAFKA config info

    Returns:
        producer object: KAFKA producer to produce messages
    """
    producer = Producer(
        {
            "bootstrap.servers": CONFIG["bootstrapServers"],
            "security.protocol": "SSL",
            "ssl.ca.location": "ca.pem",
            "ssl.certificate.location": "service.cert",
            "ssl.key.location": "service.key",
        }
    )
    return producer


def reset_offsets(
    consumer: Consumer, special_topic: str, topics: list, max_wait_time: int = 180
):
    """Set offsets to earliest for special_topic, leave others at latest."""
    # Step 1: Fetch metadata to get partition counts
    expected_partitions = {}
    start_metadata_time = time.time()
    target_offsets = {}

    while not expected_partitions and time.time() - start_metadata_time < 30:
        try:
            consumer.poll(timeout=5)
            metadata = consumer.list_topics(timeout=10)
            for topic in topics:
                if topic in metadata.topics:
                    expected_partitions[topic] = len(metadata.topics[topic].partitions)
                    logger.info(
                        f"Topic {topic} has {expected_partitions[topic]} partitions"
                    )
                else:
                    logger.warning(f"Topic {topic} not found in metadata")
            break
        except Exception as e:
            logger.warning(f"Error fetching metadata: {str(e)}")
            time.sleep(1)

    if not expected_partitions:
        logger.warning(
            "Could not fetch partition counts, proceeding without verification"
        )

    # Step 2: Wait for all partitions to be assigned
    start_time = time.time()
    while True:
        try:
            consumer.poll(timeout=5)
            assigned_partitions = consumer.assignment()

            if not assigned_partitions:
                if time.time() - start_time > max_wait_time:
                    logger.error(
                        f"Timeout: No partitions assigned after {max_wait_time}s"
                    )
                    raise Exception("Failed to get partition assignment")
                logger.info("No partitions assigned yet, polling again...")
                time.sleep(1)
                continue

            # Verify all partitions for special_topic
            assigned_by_topic = {tp.topic: [] for tp in assigned_partitions}
            for tp in assigned_partitions:
                assigned_by_topic[tp.topic].append(tp.partition)

            all_assigned = True
            for topic in topics:
                if topic in expected_partitions:
                    expected = expected_partitions[topic]
                    assigned = len(assigned_by_topic.get(topic, []))
                    if assigned < expected:
                        all_assigned = False
                        logger.info(
                            f"Topic {topic}: {assigned}/{expected} partitions assigned"
                        )

            if all_assigned:
                logger.info("All partitions assigned for all topics")
                break
            elif time.time() - start_time > max_wait_time:
                logger.warning("Timeout reached with partial assignment. Proceeding...")
                break
            else:
                elapsed = time.time() - start_time
                logger.info(
                    f"Waiting for more partitions... ({elapsed:.1f}s/{max_wait_time}s)"
                )
                time.sleep(1)

        except KafkaError as e:
            logger.error(f"Kafka error during poll: {str(e)}")
            if time.time() - start_time > max_wait_time:
                raise Exception("Failed due to Kafka errors")
            time.sleep(1)

    # Step 3: Set offsets only for special_topic
    for tp in assigned_partitions:
        if tp.topic == special_topic:
            low, high = consumer.get_watermark_offsets(tp)
            consumer.seek(TopicPartition(tp.topic, tp.partition, low))
            logger.info(
                f"Set {tp.topic} partition {tp.partition} to earliest offset: {low}, latest offset: {high}"
            )
            target_offset = high - 1
            target_offsets[(tp.topic, tp.partition)] = target_offset

        # Non-special topics remain at latest due to auto.offset.reset="latest"
    return target_offsets
