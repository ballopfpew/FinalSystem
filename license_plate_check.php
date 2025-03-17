<?php
header('Content-Type: application/json');

// --- Database Connection ---
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "projectfinal05";

$conn = new mysqli($servername, $username, $password, $dbname);
if ($conn->connect_error) {
    die(json_encode(["error" => "Database connection failed."]));
}
$conn->set_charset("utf8mb4");

// --- รับค่าจาก Python ---
$plateNumber = $_POST['plate_number'] ?? null;
$x1 = intval($_POST['x1'] ?? 0);
$y1 = intval($_POST['y1'] ?? 0);
$x2 = intval($_POST['x2'] ?? 0);
$y2 = intval($_POST['y2'] ?? 0);

// --- Debug: ตรวจสอบค่าที่ได้รับจาก Python ---
if (!$plateNumber) {
    die(json_encode(["error" => "No plate number received.", "debug" => $_POST]));
}

// --- ค้นหาป้ายทะเบียนในฐานข้อมูล ---
$sql = "SELECT plate_number, owner_name, province FROM license_plates WHERE plate_number LIKE ?";
$stmt = $conn->prepare($sql);
$searchTerm = "%$plateNumber%";
$stmt->bind_param("s", $searchTerm);
$stmt->execute();
$result = $stmt->get_result();

$data = [];
while ($row = $result->fetch_assoc()) {
    $data[] = $row;
}

$stmt->close();
$conn->close();

// --- ส่งผลลัพธ์กลับไปยัง Python ---
echo json_encode([
    "plate_number" => $plateNumber,
    "found" => !empty($data),
    "results" => $data
]);
?>
